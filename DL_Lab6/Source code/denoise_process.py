import os
import argparse
import torch
from torchvision.utils import save_image, make_grid

from dataloader import object_mapping, Convert_label
from models import ConditionalUNet
from diffusion import GaussianDiffusion


def load_checkpoint(model, checkpoint_path, device, use_ema=False):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if use_ema and "ema_state_dict" in ckpt:
        model_state = model.state_dict()
        ema_state = ckpt["ema_state_dict"]

        for name in model_state:
            if name in ema_state:
                model_state[name] = ema_state[name]

        model.load_state_dict(model_state)
        print(f"[Load] Loaded EMA weights from {checkpoint_path}")

    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] Loaded model weights from {checkpoint_path}")

    else:
        model.load_state_dict(ckpt)
        print(f"[Load] Loaded raw state_dict from {checkpoint_path}")

    return model


def build_condition(objects_json, device):
    labels = ["red sphere", "cyan cylinder", "cyan cube"]

    obj_to_idx = object_mapping(objects_json)
    condition = Convert_label(labels, obj_to_idx)

    condition = condition.unsqueeze(0).to(device)  # [1, 24]

    print("Denoising condition:", labels)
    print("Condition shape:", condition.shape)

    return condition


@torch.no_grad()
def generate_denoising_process(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    model = ConditionalUNet(image_channels=3,
                            condition_dim=24,
                            base_channels=args.base_channels,
                            time_dim=args.time_dim).to(device)

    model = load_checkpoint(model=model,
                            checkpoint_path=args.checkpoint,
                            device=device,
                            use_ema=args.use_ema)

    model.eval()

    diffusion = GaussianDiffusion(model=model,
                                  image_size=args.image_size,
                                  channels=3,
                                  timesteps=args.timesteps,
                                  beta_schedule=args.beta_schedule,
                                  loss_type="mse").to(device)

    condition = build_condition(args.objects_json, device)

    # Manually record selected denoising steps.
    record_steps = [999, 800, 600, 400, 250, 180, 120, 80, 50, 30, 10, 0]
    record_steps = set(record_steps)

    x = torch.randn(
        1,
        diffusion.channels,
        diffusion.image_size,
        diffusion.image_size,
        device=device
    )

    process_images = []

    for i in reversed(range(diffusion.timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        x = diffusion.p_sample(
            x=x,
            t=t,
            condition=condition,
            guidance_scale=args.guidance_scale
        )

        if i in record_steps:
            process_images.append(x.detach().cpu())

    process_images = torch.cat(process_images, dim=0)
    process_images = diffusion.denormalize(process_images)

    grid = make_grid(
        process_images,
        nrow=process_images.size(0),
        padding=2,
    )

    save_image(grid, args.output_path)

    print(f"[Save] Denoising process saved to: {args.output_path}")
    print("Process image shape:", process_images.shape)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_final.pt")
    parser.add_argument("--objects_json", type=str, default="file/objects.json")

    parser.add_argument("--output_path", type=str, default="images/denoising_process_12.png")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])

    parser.add_argument("--guidance_scale", type=float, default=2.6)
    parser.add_argument("--process_steps", type=int, default=8)

    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_denoising_process(args)