import torch.nn as nn
import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResBlock(nn.Module):
    def __init__(self, channel,  groups=4):     # [group 可以調大 -> 減少參數與計算量]
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, groups=groups, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, kernel_size=3, groups=groups, padding=1, bias=True)
        )
        self.relu = nn.ReLU(True)
    def forward(self, input):
        x = self.net(input)
        output = self.relu(x + input)
        return output
    
class SymmetricLinear(nn.Module):
    def __init__(self, size):
        super(SymmetricLinear, self).__init__()
        self.size = size
        self.lower_triangular = nn.Parameter(0.01*torch.randn(size, size))
        self.bias_param = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        # 建構對稱矩陣 W
        W = torch.tril(self.lower_triangular) + torch.tril(self.lower_triangular, -1).T
        out = x @ W.T
        out = out + self.bias_param
        return out
        
class Model_1(nn.Module):       
    def __init__(self, channel=172, groups=4):      # [group 可以調大 -> 減少參數與計算量]
        super(Model_1, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=groups, bias=True),
            nn.ReLU(True)
        )
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=groups, bias=True)
        self.res = ResBlock(channel)
    def forward(self, input):
        temp = self.net(input)    
        temp = self.res(self.res(self.res(temp)))

        temp = self.conv3x3(temp)
        output = input + temp
        return output                           # [output Z:M * L]

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # D : downsample
        self.sfc = SymmetricLinear(size=12)

    def forward(self, input, rho, D):
        """
        input: (batch, 172, H, W)
        D: torch.Size([12, 172])
        """
        B, _, H, W = input.shape
        
        rho = torch.clamp(rho, min=1e-4)
        Y_down = torch.einsum('bij, bjhw -> bihw', D, input)
        Y_down = Y_down.permute(0,2,3,1).reshape(-1, 12)
        Y_sfc = self.sfc(Y_down)
        N = Y_sfc.reshape(B, H, W, 12).permute(0,3,1,2)

        Y_up = torch.einsum('bij, bjhw -> bihw', D.transpose(1,2), N)

        output = (input - (2/rho)*Y_up)/rho

        return output
    
# =============== Stage ===============
class stage_1(nn.Module):
    def __init__(self):
        super(stage_1, self).__init__()
        self.M1 = Model_1(channel=172)
        self.M2 = Model_2()
        self.up = nn.Conv2d(in_channels=12, out_channels=172, kernel_size=1)

    def forward(self, YS, rho, D):
        
        YH_0 = self.up(YS)
        Z1 = self.M1(YH_0)
        U0 = torch.zeros_like(YH_0, device=device)
        """
        YS : (batch, 12, H, W)
        D : (batch, 12, 172)
        """
        DTYS = torch.einsum('bji, bjhw -> bihw', D, YS)         # DTYS : torch.Size([2, 172, 64, 64])
        # print(f'[Stage 1]: DTYS : {DTYS.shape}')            
        YH_1 = self.M2(DTYS* 2 + Z1 * rho, rho, D)
        U1 = U0 + Z1 - YH_1

        return Z1, U1, YH_1

class stage_k(nn.Module):
    def __init__(self):
        super(stage_k, self).__init__()
        self.M1 = Model_1(channel=172)
        self.M2 = Model_2()

    def forward(self, YS, Yk_1, Uk_1, rho, D):
        Zk = self.M1(Yk_1 - Uk_1)

        DTYS = torch.einsum('bji, bjhw -> bihw', D, YS)
        
        YH_k = self.M2(2*DTYS + rho*(Zk + Uk_1), rho, D)
        Uk = Uk_1 + Zk - YH_k 

        return Zk, Uk, YH_k
    
class stage_K(nn.Module):
    def __init__(self):
        super(stage_K, self).__init__()
        self.M1 = Model_1(channel=172)
    def forward(self, YK_1, UK_1):
        ZK = self.M1(YK_1 - UK_1)

        return ZK
    


