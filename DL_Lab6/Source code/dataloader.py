import json
import os
import torch

from torch.utils.data import DataLoader, Dataset
from  typing import Dict, List, Tuple
from PIL import Image
from torchvision import transforms

def object_mapping(object_path):
    with open(object_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if all(isinstance(v, int) for v in obj.values()):
        obj_index = obj
    elif all (isinstance(v, str) for v in obj.values()):
        obj_index = {label: int(idx) for idx, label in obj.items()}
    else:
        raise ValueError("What's wrong with you?")
    
    return obj_index

def Convert_label(labels:List[str], obj_index: Dict[str, int]):
    condition = torch.zeros(len(obj_index), dtype=torch.float32)

    for label in labels:
        condition[obj_index[label]] = 1.0

    return condition

class ICLEVRDataset(Dataset):
    def __init__(self, image_dir, json_path, object_json_path, image_size=64):
        super().__init__()

        self.image_dir = image_dir
        self.json_path = json_path
        self.obj_to_idx = object_mapping(object_json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            self.annotions = json.load(f)

        self.filenames = list(self.annotions.keys())
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        labels = self.annotions[filename]

        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        condition = Convert_label(labels, self.obj_to_idx)

        return {"image": image, "condition": condition, "filename": filename}
    
class ICLEVRTestConditionDataset(Dataset):
    def __init__(self, json_path, object_json_path):
            super().__init__()

            self.json_path = json_path
            self.object_to_idx = object_mapping(object_json_path)

            with open(json_path, "r", encoding="utf-8") as f:
                self.conditions = json.load(f)

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, idx: int):
        labels = self.conditions[idx]
        condition = Convert_label(labels, self.object_to_idx)

        return {"condition": condition, "index": idx}
    
def get_train_loader(
    image_dir, train_json_path, object_json_path, image_size=64, batch_size=64, num_workers=4, shuffle=True):

    dataset = ICLEVRDataset(image_dir=image_dir, json_path=train_json_path, object_json_path=object_json_path, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader

def get_test_condition_loader(test_json_path, object_json_path, batch_size=32, num_workers=0):
    
    dataset = ICLEVRTestConditionDataset(json_path=test_json_path, object_json_path=object_json_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader

if __name__ == "__main__":
    image_dir = "iclevr"

    train_json_path = "file/train.json"
    object_json_path = "file/objects.json"

    train_loader = get_train_loader(image_dir=image_dir, train_json_path=train_json_path, object_json_path=object_json_path, image_size=64, batch_size=8, num_workers=0)

    batch = next(iter(train_loader))

    print("Image shape:", batch["image"].shape)
    print("Condition shape:", batch["condition"].shape)
    # print("Example labels:", batch["labels"][0])
    print("Example filename:", batch["filename"][0])
    print("Condition vector:", batch["condition"][0])