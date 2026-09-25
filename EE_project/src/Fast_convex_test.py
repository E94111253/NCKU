import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiplicativeCNMFTest(nn.Module):
    """
    Test version of coupled-NMF for COS2A.

    Purpose:
      - More stable than the previous gradient-descent fast_convex.py.
      - Uses multiplicative NMF-style updates for A and S.
      - Keeps non-negativity naturally.
      - Tries to preserve Y_DE, while also fitting the high-resolution Sentinel-2 bands.

    Inputs:
      Y_de : (B, 172, H, W)  deep rough HSI
      Y_ms : (B, M,   H, W)  high-resolution MSI bands, usually M=4
      D_e  : (B, M, 172)     scene-adaptive spectral response

    Outputs:
      A : (B, 172, R)
      S : (B, R, H*W)
    """

    def __init__(
        self,
        rank=10,
        iters=80,
        w_de=1.0,
        w_ms=0.35,
        lambda_sparse=1e-5,
        ridge=1e-3,
        s_max=3.0,
        a_max=1.0,
        print_every=20,
        eps=1e-8,
    ):
        super().__init__()
        self.rank = rank
        self.iters = iters
        self.w_de = w_de
        self.w_ms = w_ms
        self.lambda_sparse = lambda_sparse
        self.ridge = ridge
        self.s_max = s_max
        self.a_max = a_max
        self.print_every = print_every
        self.eps = eps

    @staticmethod
    def _forward_flat(A, S):
        # A: (B,C,R), S: (B,R,N) -> (B,C,N)
        return torch.bmm(A, S)

    @staticmethod
    def _select_endmembers_farthest(Y_flat, rank):
        """
        Y_flat: (B,C,N). Pick rank spectra from pixels by farthest-point sampling.
        Returns A: (B,C,R)
        """
        B, C, N = Y_flat.shape
        A_list = []
        for b in range(B):
            Y = Y_flat[b].transpose(0, 1).contiguous()  # (N,C)
            Y_norm = Y / (torch.norm(Y, dim=1, keepdim=True) + 1e-8)

            energy = Y.mean(dim=1)
            first = torch.argmax(energy).item()
            selected = [first]
            min_dist = torch.full((N,), float("inf"), device=Y.device)

            for _ in range(1, rank):
                last = Y_norm[selected[-1]].unsqueeze(0)
                dist = torch.sum((Y_norm - last) ** 2, dim=1)
                min_dist = torch.minimum(min_dist, dist)
                nxt = torch.argmax(min_dist).item()
                selected.append(nxt)

            idx = torch.tensor(selected, device=Y_flat.device, dtype=torch.long)
            A_list.append(Y_flat[b:b + 1, :, idx])

        A = torch.cat(A_list, dim=0).contiguous()
        return torch.clamp(A, min=1e-6, max=1.0)

    def _init_s_ridge(self, A, Y_flat):
        """
        S = argmin ||AS - Y||^2 + ridge ||S||^2, then ReLU.
        A: (B,C,R), Y_flat: (B,C,N)
        """
        B, C, R = A.shape
        At = A.transpose(1, 2)                    # (B,R,C)
        AtA = torch.bmm(At, A)                    # (B,R,R)
        eye = torch.eye(R, device=A.device).unsqueeze(0)
        AtA = AtA + self.ridge * eye
        AtY = torch.bmm(At, Y_flat)               # (B,R,N)
        S = torch.linalg.solve(AtA, AtY)
        S = torch.clamp(S, min=0.0, max=self.s_max)
        return S.contiguous()

    def _scale_s_to_y(self, A, S, Y_flat):
        """Global least-squares scale for S so AS matches Y_de magnitude."""
        with torch.no_grad():
            X = self._forward_flat(A, S)
            num = (X * Y_flat).sum(dim=(1, 2), keepdim=True)
            den = (X * X).sum(dim=(1, 2), keepdim=True) + self.eps
            scale = num / den
            S = S * scale
            S = torch.clamp(S, min=0.0, max=self.s_max)
        return S

    def _stabilize_scale(self, A, S):
        """
        Keep A in a reflectance-like scale and absorb the inverse scale into S.
        Only rescales columns whose max exceeds a_max. This avoids A exploding.
        """
        with torch.no_grad():
            col_max = A.amax(dim=1, keepdim=True).clamp_min(self.eps)  # (B,1,R)
            scale = torch.clamp(col_max / self.a_max, min=1.0)        # only downscale if too large
            A = A / scale
            S = S * scale.squeeze(1).unsqueeze(-1)
            A = torch.clamp(A, min=1e-8, max=self.a_max)
            S = torch.clamp(S, min=0.0, max=self.s_max)
        return A, S

    def forward(self, Y_de, Y_ms, D_e):
        B, C, H, W = Y_de.shape
        _, M, Hm, Wm = Y_ms.shape
        assert C == 172, f"Expected Y_de channel=172, got {C}"
        assert H == Hm and W == Wm, "Y_de and Y_ms must have same spatial size"
        assert D_e.shape == (B, M, C), f"Expected D_e shape {(B, M, C)}, got {tuple(D_e.shape)}"

        Y_de = torch.clamp(Y_de.detach(), 0.0, 1.0)
        Y_ms = torch.clamp(Y_ms.detach(), 0.0, 1.0)
        D_e = torch.clamp(D_e.detach(), min=0.0)
        D_e = D_e / (D_e.sum(dim=2, keepdim=True) + self.eps)

        N = H * W
        Y_flat = Y_de.reshape(B, C, N)
        M_flat = Y_ms.reshape(B, M, N)

        # ----- initialization -----
        A = self._select_endmembers_farthest(Y_flat, self.rank)
        S = self._init_s_ridge(A, Y_flat)
        S = self._scale_s_to_y(A, S, Y_flat)
        A, S = self._stabilize_scale(A, S)

        best_obj = float("inf")
        best_A = A.clone()
        best_S = S.clone()

        for k in range(self.iters):
            # =============================
            # Update S multiplicatively
            # =============================
            DA = torch.bmm(D_e, A)                       # (B,M,R)
            AtY = torch.bmm(A.transpose(1, 2), Y_flat)   # (B,R,N)
            DAtM = torch.bmm(DA.transpose(1, 2), M_flat) # (B,R,N)

            AtA = torch.bmm(A.transpose(1, 2), A)        # (B,R,R)
            DAtDA = torch.bmm(DA.transpose(1, 2), DA)    # (B,R,R)

            num_S = self.w_de * AtY + self.w_ms * DAtM
            den_S = (
                self.w_de * torch.bmm(AtA, S)
                + self.w_ms * torch.bmm(DAtDA, S)
                + self.lambda_sparse
                + self.eps
            )
            S = S * (num_S / den_S)
            S = torch.clamp(S, min=0.0, max=self.s_max)

            # =============================
            # Update A multiplicatively
            # =============================
            SS = torch.bmm(S, S.transpose(1, 2))         # (B,R,R)
            YSt = torch.bmm(Y_flat, S.transpose(1, 2))   # (B,C,R)
            MSt = torch.bmm(M_flat, S.transpose(1, 2))   # (B,M,R)

            DtMSt = torch.bmm(D_e.transpose(1, 2), MSt)  # (B,C,R)
            DtD = torch.bmm(D_e.transpose(1, 2), D_e)    # (B,C,C)

            num_A = self.w_de * YSt + self.w_ms * DtMSt
            den_A = (
                self.w_de * torch.bmm(A, SS)
                + self.w_ms * torch.bmm(torch.bmm(DtD, A), SS)
                + self.eps
            )
            A = A * (num_A / den_A)
            A = torch.clamp(A, min=1e-8, max=10.0)

            A, S = self._stabilize_scale(A, S)

            # ----- monitor objective -----
            with torch.no_grad():
                X = self._forward_flat(A, S)             # (B,C,N)
                MX = torch.bmm(D_e, X)                   # (B,M,N)
                term_de = F.mse_loss(X, Y_flat)
                term_ms = F.mse_loss(MX, M_flat)
                obj = self.w_de * term_de + self.w_ms * term_ms + self.lambda_sparse * S.mean()

                if obj.item() < best_obj:
                    best_obj = obj.item()
                    best_A = A.clone()
                    best_S = S.clone()

                if self.print_every is not None and k % self.print_every == 0:
                    print(
                        f"iter={k:03d} "
                        f"term_de={term_de.item():.6f} "
                        f"term_ms={term_ms.item():.6f} "
                        f"obj={obj.item():.6f} "
                        f"Amax={A.max().item():.4f} "
                        f"Smax={S.max().item():.4f} "
                        f"Sstd={S.std().item():.6f}"
                    )

        return best_A.detach(), best_S.detach()


# Backward-compatible class name for your algo.py
class fast_convex_(MultiplicativeCNMFTest):
    def __init__(self, r=10, iters=80, lam=0.1, blur_r=2):
        # lam is kept for compatibility; here it maps weakly to lambda_sparse.
        super().__init__(
            rank=r,
            iters=iters,
            w_de=1.0,
            w_ms=0.35,
            lambda_sparse=1e-5,
            ridge=1e-3,
            s_max=3.0,
            a_max=1.0,
            print_every=20,
        )
