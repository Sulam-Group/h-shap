import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

data_dir = os.path.join("data")
explanation_dir = os.path.join("explanations")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform)
dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, num_workers=0
)
_iter = iter(dataloader)
X, _ = next(_iter)
ref = X.detach().mean(0)

torch.save(ref, os.path.join(explanation_dir, "reference.pt"))
