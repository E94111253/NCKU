import torch
import torch.nn as nn
import torch.nn.functional as F


class CoupledCNMF(nn.Module):
    """
    Stable coupled NMF module for COS2A-style post-processing.

    It solves an approximate non-negative factorization problem:

        min_{A,S >= 0}  w_de ||Y_de - A S||_F^2
                      + w_ms ||Y_ms - D A S||_F^2
                      + lambda_sparse * mean(S)

    Shapes:
        Y_de : (B, 172, H, W)      deep rough hyperspectral image
        Y_ms : (B, M,   H, W)      Sentinel-2 bands used as high-resolution constraint, usually M=4
        D    : (B, M, 172)         adaptive spectral response matrix
        A    : (B, 172, R)         spectral basis / endmembers
        S    : (B, R, H*W)         abundance map flattened spatially
    """

    def __init__(self, rank=10, iters=50, w_de=0.5, w_ms=1.0, lambda_sparse=1e-5, ridge=1e-3, a_max=1.0,  s_max=3.0, print_every=None, eps=1e-8):
        super().__init__()
        self.rank = rank
        self.iters = iters
        self.w_de = w_de
        self.w_ms = w_ms
        self.lambda_sparse = lambda_sparse
        self.ridge = ridge
        self.a_max = a_max
        self.s_max = s_max
        self.print_every = print_every
        self.eps = eps

    @staticmethod
    def _flat(x):
        # (B,C,H,W) -> (B,C,N)
        return x.reshape(x.shape[0], x.shape[1], -1)

    @staticmethod
    def _reconstruct(A, S):
        # A: (B,C,R), S: (B,R,N) -> (B,C,N)
        return torch.bmm(A, S)

    def _init_A_farthest(self, Y):
        """
        Initialize A using farthest-point sampling from Y_de spectra.
        This is much stabler than random initialization.
        Y: (B,C,N)
        """
        B, C, N = Y.shape
        A_all = []

        for b in range(B):
            spectra = Y[b].transpose(0, 1).contiguous()  # (N,C)
            spectra_n = spectra / (torch.norm(spectra, dim=1, keepdim=True) + self.eps)

            # Start from the brightest pixel, then choose farthest spectra.
            first = torch.argmax(spectra.mean(dim=1)).item()
            selected = [first]
            min_dist = torch.full((N,), float("inf"), device=Y.device)

            for _ in range(1, self.rank):
                last = spectra_n[selected[-1]].unsqueeze(0)
                dist = torch.sum((spectra_n - last) ** 2, dim=1)
                min_dist = torch.minimum(min_dist, dist)
                selected.append(torch.argmax(min_dist).item())

            idx = torch.tensor(selected, device=Y.device, dtype=torch.long)
            A_all.append(Y[b:b + 1, :, idx])

        A = torch.cat(A_all, dim=0)
        return torch.clamp(A, min=self.eps, max=self.a_max)

    def _init_S_ridge(self, A, Y):
        """
        Initialize S by non-negative ridge least squares:
            S = (A'A + ridge I)^(-1) A'Y
        """
        B, C, R = A.shape
        At = A.transpose(1, 2)                     # (B,R,C)
        AtA = torch.bmm(At, A)                     # (B,R,R)
        I = torch.eye(R, device=A.device, dtype=A.dtype).unsqueeze(0)
        AtY = torch.bmm(At, Y)                     # (B,R,N)
        S = torch.linalg.solve(AtA + self.ridge * I, AtY)
        return torch.clamp(S, min=0.0, max=self.s_max)

    def _stabilize_scale(self, A, S):
        """
        NMF has scale ambiguity: A*S = (A/c)*(cS).
        Keep A reflectance-like and absorb the scale into S.
        """
        col_max = A.amax(dim=1, keepdim=True).clamp_min(self.eps)   # (B,1,R)
        scale = torch.clamp(col_max / self.a_max, min=1.0)
        A = A / scale
        S = S * scale.squeeze(1).unsqueeze(-1)
        A = torch.clamp(A, min=self.eps, max=self.a_max)
        S = torch.clamp(S, min=0.0, max=self.s_max)
        return A, S

    def _objective(self, A, S, Y, M, D):
        X = self._reconstruct(A, S)
        MX = torch.bmm(D, X)
        term_de = F.mse_loss(X, Y)
        term_ms = F.mse_loss(MX, M)
        obj = self.w_de * term_de + self.w_ms * term_ms + self.lambda_sparse * S.mean()
        return obj, term_de, term_ms

    @torch.no_grad()
    def forward(self, Y_de, Y_ms, D):
        B, C, H, W = Y_de.shape
        _, M, Hm, Wm = Y_ms.shape

        assert C == 172, f"Expected Y_de with 172 bands, got {C}"
        assert H == Hm and W == Wm, "Y_de and Y_ms must have the same H,W"
        assert D.shape == (B, M, C), f"Expected D shape {(B, M, C)}, got {tuple(D.shape)}"

        Y_de = torch.clamp(Y_de, 0.0, 1.0)
        Y_ms = torch.clamp(Y_ms, 0.0, 1.0)
        D = torch.clamp(D, min=0.0)
        D = D / (D.sum(dim=2, keepdim=True) + self.eps)

        Y = self._flat(Y_de)   # (B,172,N)
        M_flat = self._flat(Y_ms)  # (B,M,N)

        A = self._init_A_farthest(Y)
        S = self._init_S_ridge(A, Y)
        A, S = self._stabilize_scale(A, S)

        best_A = A.clone()
        best_S = S.clone()
        best_obj = float("inf")

        for k in range(self.iters):
            DA = torch.bmm(D, A)  # (B,M,R)

            # ----- update S -----
            AtY = torch.bmm(A.transpose(1, 2), Y)
            DAtM = torch.bmm(DA.transpose(1, 2), M_flat)
            AtA = torch.bmm(A.transpose(1, 2), A)
            DAtDA = torch.bmm(DA.transpose(1, 2), DA)

            num_S = self.w_de * AtY + self.w_ms * DAtM
            den_S = (
                self.w_de * torch.bmm(AtA, S)
                + self.w_ms * torch.bmm(DAtDA, S)
                + self.lambda_sparse
                + self.eps
            )
            S = S * (num_S / den_S)
            S = torch.clamp(S, min=0.0, max=self.s_max)

            # ----- update A -----
            SS = torch.bmm(S, S.transpose(1, 2))
            YSt = torch.bmm(Y, S.transpose(1, 2))
            MSt = torch.bmm(M_flat, S.transpose(1, 2))
            DtMSt = torch.bmm(D.transpose(1, 2), MSt)
            DtD = torch.bmm(D.transpose(1, 2), D)

            num_A = self.w_de * YSt + self.w_ms * DtMSt
            den_A = (
                self.w_de * torch.bmm(A, SS)
                + self.w_ms * torch.bmm(torch.bmm(DtD, A), SS)
                + self.eps
            )
            A = A * (num_A / den_A)
            A = torch.clamp(A, min=self.eps, max=10.0)
            A, S = self._stabilize_scale(A, S)

            obj, term_de, term_ms = self._objective(A, S, Y, M_flat, D)
            if obj.item() < best_obj:
                best_obj = obj.item()
                best_A = A.clone()
                best_S = S.clone()

            if self.print_every is not None and k % self.print_every == 0:
                print(f"iter={k:03d} "
                    f"term_de={term_de.item():.6f} "
                    f"term_ms={term_ms.item():.6f} "
                    f"obj={obj.item():.6f} "
                    f"Amax={A.max().item():.4f} "
                    f"Smax={S.max().item():.4f} "
                    f"Sstd={S.std().item():.6f}")

        return best_A, best_S


class fast_convex_(CoupledCNMF):
    def __init__(self, r=6, iters=50, lam=0.1, blur_r=2, print_every=None):
        # blur_r is kept only for compatibility with your old call.
        super().__init__(rank=r, iters=iters, w_de=1.5, w_ms=0.5, lambda_sparse=1e-5, ridge=1e-3, a_max=1.0, s_max=3.0, print_every=print_every)
