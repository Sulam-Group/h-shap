import os
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")

ops = ["train", "val", "test"]
batch_size = 64
t = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
datasets = {
    op: ImageFolder(
        os.path.join(data_dir, op),
        transform=t,
    )
    for op in ops
}
dataloaders = {
    op: DataLoader(
        d,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    for op, d in datasets.items()
}

model = Net()
model = model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

ops = ["train", "val"]
num_epochs = 4
for epoch in range(num_epochs):
    print(f"Started epoch {epoch + 1}")
    for op in ops:

        if op == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        running_loss = 0.0
        running_corrects = 0

        dataloader = dataloaders[op]
        for i, data in enumerate(tqdm(dataloader)):
            input, label = data

            input = input.to(device)
            label = label.to(device)

            output = model(input)
            loss = criterion(output, label)
            prediction = output.argmax(dim=1)

            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(prediction == label)

            if op == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_loss = running_loss / len(datasets[op])
        epoch_accuracy = running_corrects / len(datasets[op])
        print(f"{op} loss: {epoch_loss:.4f}")
        print(f"{op} accuracy: {epoch_accuracy:.4f}")

op = "test"
model.eval()
torch.set_grad_enabled(False)

dataloader = dataloaders[op]
running_corrects = 0
for i, data in enumerate(tqdm(dataloader)):
    input, label = data
    label = (label >= 1).long()

    input = input.to(device)
    label = label.to(device)

    output = model(input)
    prediction = output.argmax(dim=1)

    running_corrects += torch.sum(prediction == label)

test_accuracy = running_corrects / len(datasets[op])
print(f"{op} accuracy: {test_accuracy:.4f}")
torch.save(model.state_dict(), os.path.join(model_dir, "_model.pt"))
