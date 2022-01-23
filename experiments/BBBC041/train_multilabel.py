import os
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, split, transform):
        self.data_dir = data_dir
        self.split = split
        with open(os.path.join(self.data_dir, self.split), "rb") as f:
            self.data = pickle.load(f)
        self.transform = transform
        self.image_dir = "/export/gaon1/data/jteneggi/data/malaria/cropped_images"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, label = self.data[idx]

        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        return image, torch.tensor(label).float()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = os.path.join("data")
multilabel_dir = os.path.join(data_dir, "multilabel")

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

ops = ["train", "test"]
datasets = {
    phase: MultiLabelDataset(data_dir, phase, data_transforms[phase])
    for phase in ["train", "test"]
}
dataloaders = {
    phase: DataLoader(datasets[phase], batch_size=4, shuffle=True, num_workers=6)
    for phase in ["train", "test"]
}
dataset_sizes = {phase: len(datasets[phase]) for phase in ["train", "test"]}
