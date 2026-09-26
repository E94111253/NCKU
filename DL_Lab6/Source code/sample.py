import os
import argparse
import torch
from torchvision.utils import save_image, make_grid

from dataloader import get_test_condition_loader
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
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] Loaded model weights from {checkpoint_path}")

    return model


@torch.no_grad()
def generate_images(args, json_path, output_dir, grid_path):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)

    loader = get_test_condition_loader(test_json_path=json_path,
                                       object_json_path=args.objects_json,
                                       batch_size=args.batch_size,
                                       num_workers=0)

    model = ConditionalUNet(image_channels=3,
                            condition_dim=24,
                            base_channels=args.base_channels,
                            time_dim=args.time_dim).to(device)

    model = load_checkpoint(model, args.checkpoint, device, use_ema=args.use_ema)
    model.eval()

    diffusion = GaussianDiffusion(model=model,
                                  image_size=args.image_size,
                                  channels=3,
                                  timesteps=args.timesteps,
                                  beta_schedule=args.beta_schedule,
                                  loss_type="mse").to(device)

    all_images = []

    for batch in loader:
        conditions = batch["condition"].to(device)
        indices = batch["index"]

        samples = diffusion.sample(
            condition=conditions,
            batch_size=conditions.size(0),
            guidance_scale=args.guidance_scale)

        samples = diffusion.denormalize(samples)

        for img, idx in zip(samples, indices):
            save_path = os.path.join(output_dir, f"{int(idx)}.png")
            save_image(img, save_path)
            all_images.append(img.cpu())

    all_images = torch.stack(all_images, dim=0)

    grid = make_grid(all_images, nrow=8, padding=2)

    save_image(grid, grid_path)

    print(f"[Save] Generated images saved to: {output_dir}")
    print(f"[Save] Grid saved to: {grid_path}")
    print(f"Generated image shape: {all_images.shape}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="checkpoints/ddpm_final.pt")
    parser.add_argument("--objects_json", type=str, default="file/objects.json")

    parser.add_argument("--test_json", type=str, default="file/test.json")
    parser.add_argument("--new_test_json", type=str, default="file/new_test.json")

    parser.add_argument("--output_test_dir", type=str, default="images_2.6/test")
    parser.add_argument("--output_new_test_dir", type=str, default="images_2.6/new_test")

    parser.add_argument("--grid_test_path", type=str, default="images_2.6/test_grid.png")
    parser.add_argument("--grid_new_test_path", type=str, default="images_2.6/new_test_grid.png")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--guidance_scale", type=float, default=2.6)
    parser.add_argument("--use_ema", action="store_true")

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Generating images for test.json...")
    generate_images(args=args,
                    json_path=args.test_json,
                    output_dir=args.output_test_dir,
                    grid_path=args.grid_test_path)

    print("Generating images for new_test.json...")
    generate_images(args=args,
                    json_path=args.new_test_json,
                    output_dir=args.output_new_test_dir,
                    grid_path=args.grid_new_test_path)

    print("Sampling finished.")