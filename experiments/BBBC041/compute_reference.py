import os
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms

data_dir = os.path.join("data")
trophozoite_dir = os.path.join(data_dir, "trophozoite")
explanation_dir = os.path.join("explanations")
figure_dir = os.path.join("figures")
os.makedirs(figure_dir, exist_ok=True)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)
unnorm = transforms.Normalize(-mean / std, 1 / std)

train_dataset = ImageFolder(os.path.join(trophozoite_dir, "train"), transform)
dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, num_workers=0
)
_iter = iter(dataloader)
X, _ = next(_iter)
ref = X.detach().mean(0)

torch.save(ref, os.path.join(explanation_dir, "reference.pt"))

ref = unnorm(ref).permute(1, 2, 0).numpy()
plt.imsave(os.path.join(figure_dir, "reference.jpg"), ref)
