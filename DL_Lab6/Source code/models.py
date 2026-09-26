import math
import torch
import torch.nn as nn

class embedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim//2

        emb_scale = math.log(10000)/(half_dim-1)
        emb = torch.exp(torch.arange(half_dim, device=device)*(-emb_scale))
        emb = t[:, None].float()*emb[None, :]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channel)

        self.emb_proj = nn.Linear(emb_dim, out_channel)

        if in_channel != out_channel:
            self.shortcut = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))

        emb_out = 0.3 * torch.tanh(self.emb_proj(emb))
        # emb_out = self.emb_proj(emb)                      # use this if learning rate is slow
        h = h + emb_out[:, :, None, None]

        h = self.act(self.norm2(self.conv2(h)))

        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        b, c, h, w = x.shape

        residual = x
        x = self.norm(x)

        x = x.view(b, c, h * w).permute(0, 2, 1)  # [B, HW, C]
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x + residual
    
class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=3, condition_dim=24, base_channels=64, time_dim=256):
        super().__init__()

        self.image_channels = image_channels
        self.condition_dim = condition_dim
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_embedding = nn.Sequential(embedding(time_dim), 
                                            nn.Linear(time_dim, time_dim),
                                            nn.SiLU(),
                                            nn.Linear(time_dim, time_dim))
        
        self.condition_embedding = nn.Sequential(nn.Linear(condition_dim, time_dim),
                                                 nn.SiLU(),
                                                 nn.Linear(time_dim, time_dim))

        emb_dim = time_dim
        self.enc1 = ResBlock(image_channels, base_channels, emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)

        self.enc2 = ResBlock(base_channels, base_channels * 2, emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.enc3 = ResBlock(base_channels * 2, base_channels * 4, emb_dim)

        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, emb_dim)
        self.attn_mid = AttentionBlock(base_channels * 4, num_heads=4)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, emb_dim)

        # Decoder: 16x16 -> 32x32 -> 64x64
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1,)
        self.dec1 = ResBlock(base_channels * 4, base_channels * 2, emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(base_channels * 2, base_channels, emb_dim)

        self.out = nn.Sequential(nn.GroupNorm(8, base_channels), 
                                 nn.SiLU(), 
                                 nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1))
        
    def forward(self, x, t, condition):
        t_emb = self.time_embedding(t)
        c_emb = self.condition_embedding(condition)

        emb = torch.tanh(t_emb + c_emb)     # stable(temp)
        # emb = t_emb + c_emb

        h1 = self.enc1(x, emb)
        x = self.down1(h1)

        h2 = self.enc2(x, emb)
        x = self.down2(h2)

        h3 = self.enc3(x, emb)

        x = self.mid1(h3, emb)
        x = self.attn_mid(x)
        x = self.mid2(x, emb)

        x = self.up1(x)                 
        x = torch.cat([x, h2], dim=1)   
        x = self.dec1(x, emb)           

        x = self.up2(x)                 
        x = torch.cat([x, h1], dim=1)   
        x = self.dec2(x, emb)         

        return self.out(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConditionalUNet(image_channels=3, condition_dim=24, base_channels=64, time_dim=256).to(device)

    x = torch.randn(8, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (8,)).to(device)
    condition = torch.randn(8, 24).to(device)

    with torch.no_grad():
        pred_noise = model(x, t, condition)

    print("Device:", device)
    print("Input image shape:", x.shape)
    print("Timestep shape:", t.shape)
    print("Condition shape:", condition.shape)
    print("Predicted noise shape:", pred_noise.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)