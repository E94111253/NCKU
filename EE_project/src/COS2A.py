import torch.nn as nn
import torch
import torch.nn.functional as F

from src.models import stage_1, stage_k, stage_K, Model_1, Model_2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class COS2A(nn.Module):
    def __init__(self, D_init):
        super(COS2A, self).__init__()
        self.S1 = stage_1()
        self.Sk = stage_k()
        self.SK = stage_K()

        shared_M1 = Model_1()
        shared_M2 = Model_2()

        self.S1.M1 = shared_M1
        self.Sk.M1 = shared_M1
        self.SK.M1 = shared_M1  

        self.S1.M2 = shared_M2
        self.Sk.M2 = shared_M2

        self.register_buffer("D", D_init.float())
        self.rho = nn.Parameter(torch.tensor(0.0))

    def forward(self, YS):
        # YS : (batch, 12, H, W)
        # ZK : (batch, 172, H, W)  
        batch = YS.shape[0]
        device = YS.device
        rho = F.softplus(self.rho) + 1e-4
        D = self.D.to(device)
        D = D.unsqueeze(0).expand(batch, -1, -1)

        Z1, U1, YH_1 = self.S1(YS, rho, D)
        Zk, Uk, YH_k = self.Sk(YS, YH_1, U1, rho, D)
        Zk, Uk, YH_k = self.Sk(YS, YH_k, Uk, rho, D)
        ZK = self.SK(YH_k, Uk)              # YDE

        return ZK                           # Y_DE: torch.Size([2, 172, 64, 64])
    

if __name__ == '__main__':
    in_channels = 12    # MSI channels
    out_channels = 172  # HSI channels
    Batch = 2
    Ys = torch.randn(Batch, 12, 64, 64, device=device)
    def build_fixed_D():
        D = torch.rand(12, 172)
        D = torch.clamp(D, min=0.0)
        D = D / (D.sum(dim=1, keepdim=True) + 1e-8)
        return D
    D_init = build_fixed_D()
    model = COS2A(D_init).to(device)
    
    Z_final = model(Ys)
    
    print(f"Input MSI shape: {Ys.shape}")
    print(f"Output HSI shape: {Z_final.shape}")
    print(f"D shape: {model.D.shape}")
    print(f"D is trainable: {model.D.requires_grad}")
    
    # Verify forward pass
    print("\nForward pass successful!")