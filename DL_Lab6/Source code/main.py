import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as  np
import argparse
import random

from tqdm import tqdm
from dataloader import get_train_loader
from models import ConditionalUNet
from diffusion import GaussianDiffusion

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data
                )

    def apply_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def save_checkpoint(model, diffusion, optimizer, epoch, step, loss, save_path, ema=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {"model_state_dict":model.state_dict(),
                  "diffusion":diffusion.state_dict(),
                  "optimizer_state_dict":optimizer.state_dict(),
                  "epoch":epoch,
                  "step":step,
                  "loss":loss}
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
    torch.save(checkpoint, save_path)
    print(f"[Save] Checkpoint saved to {save_path}")

def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    train_loader = get_train_loader(image_dir=args.image_dir,
                                    train_json_path=args.train_json,
                                    object_json_path=args.objects_json,
                                    image_size=args.image_size,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=True)
    
    model = ConditionalUNet(image_channels=3,
                            condition_dim=24,
                            base_channels=args.base_channels,
                            time_dim=args.time_dim).to(device)
    diffusion = GaussianDiffusion(model=model,
                                  image_size=args.image_size,
                                  channels=3,
                                  timesteps=args.timesteps,
                                  beta_schedule=args.beta_schedule,
                                  loss_type=args.loss_type).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    start_epoch = 1
    global_step = 0

    if args.resume is not None and os.path.exists(args.resume):
        print(f"[Resume] Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]

        print(f"[Resume] Start from epoch {start_epoch}, step {global_step}")

    os.makedirs(args.save_dir, exist_ok=True)

    model.train()
    diffusion.train()

    print("=== Start training ===")

    start_time =  time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]",leave=True,)

        for batch in progress_bar:
            images = batch["image"].to(device)          
            conditions = batch["condition"].to(device)  

            batch_size = images.size(0)
            t = torch.randint(low=0, high=args.timesteps, size=(batch_size,), device=device).long()

            loss = diffusion.p_losses(x_start=images, t=t, condition=conditions, cond_drop_prob=args.cond_drop_prob)
            
            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            if ema is not None:
                ema.update(model)

            global_step += 1
            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                save_path = os.path.join(args.save_dir, f"ddpm_step_{global_step}.pt")
                save_checkpoint(model=model,
                                diffusion=diffusion,
                                optimizer=optimizer,
                                epoch=epoch,
                                step=global_step,
                                loss=loss.item(),
                                save_path=save_path,
                                ema=ema)
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch}/{args.epochs}] "
            f"Average Loss: {avg_loss:.6f} "
            f"Elapsed: {elapsed / 60:.2f} min")
        if epoch % args.save_every_epochs == 0:
                save_path = os.path.join(args.save_dir, f"ddpm_epoch_{epoch}.pt")
                save_checkpoint(model=model,
                                diffusion=diffusion,
                                optimizer=optimizer,
                                epoch=epoch,
                                step=global_step,
                                loss=avg_loss,
                                save_path=save_path, 
                                ema=ema)

    final_path = os.path.join(args.save_dir, "ddpm_final.pt")
    save_checkpoint(model=model,
                    diffusion=diffusion,
                    optimizer=optimizer,
                    epoch=args.epochs,
                    step=global_step,
                    loss=avg_loss,
                    save_path=final_path,
                    ema=ema)

    print("Training finished.")
    print(f"Final checkpoint: {final_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="iclevr")
    parser.add_argument("--train_json", type=str, default="file/train.json")
    parser.add_argument("--objects_json", type=str, default="file/objects.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="checkpoints/ddpm_final.pt")
    parser.add_argument("--epochs", type=int, default=200)                      
    parser.add_argument("--batch_size", type=int, default=64)                   
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--save_every_epochs", type=int, default=10)
    parser.add_argument("--save_every_steps", type=int, default=0)

    opt = parser.parse_args()
    
    train(opt)