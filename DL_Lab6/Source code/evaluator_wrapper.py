import os
import json
import argparse
import sys

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from file.evaluator import evaluation_model
from dataloader import object_mapping, Convert_label


def load_conditions(json_path, objects_json):
    obj_to_idx = object_mapping(objects_json)

    with open(json_path, "r", encoding="utf-8") as f:
        conditions = json.load(f)

    labels = []
    for item in conditions:
        labels.append(Convert_label(item, obj_to_idx))

    labels = torch.stack(labels, dim=0)
    return labels


def load_generated_images(image_dir, num_images=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    images = []

    for i in range(num_images):
        image_path = os.path.join(image_dir, f"{i}.png")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing generated image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)

    images = torch.stack(images, dim=0)
    return images


def evaluate_split(name, image_dir, json_path, objects_json):
    print(f"Evaluating {name}...")

    images = load_generated_images(image_dir=image_dir, num_images=32)
    labels = load_conditions(json_path=json_path, objects_json=objects_json)

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    evaluator = evaluation_model()

    images = images.cuda()
    labels = labels.cuda()

    acc = evaluator.eval(images, labels)

    print("=" * 50)
    print(f"{name} Accuracy: {acc:.4f}")
    print("=" * 50)

    return acc


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--objects_json", type=str, default="file/objects.json")

    parser.add_argument("--test_json", type=str, default="file/test.json")
    parser.add_argument("--new_test_json", type=str, default="file/new_test.json")

    parser.add_argument("--test_image_dir", type=str, default="images/test")
    parser.add_argument("--new_test_image_dir", type=str, default="images/new_test")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_acc = evaluate_split(
        name="test.json",
        image_dir=args.test_image_dir,
        json_path=args.test_json,
        objects_json=args.objects_json,
    )

    new_test_acc = evaluate_split(
        name="new_test.json",
        image_dir=args.new_test_image_dir,
        json_path=args.new_test_json,
        objects_json=args.objects_json,
    )

    print("Final Results")
    print(f"test.json accuracy:     {test_acc:.4f}")
    print(f"new_test.json accuracy: {new_test_acc:.4f}")
