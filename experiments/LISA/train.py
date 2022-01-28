import os
import torch
import torch.nn as nn
import copy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")
os.makedirs(model_dir, exist_ok=True)

ops = ["train", "val"]
batch_size = 4
t = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}
datasets = {
    op: ImageFolder(
        os.path.join(data_dir, op),
        t[op],
    )
    for op in ops
}
dataloaders = {
    op: DataLoader(
        d,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    for op, d in datasets.items()
}

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

num_epochs = 25
best_accuracy = 0.0
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

        if op == "train":
            scheduler.step()

        if op == "val" and epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model_state_dict = copy.deepcopy(model.state_dict())

print(f"Best accuracy: {best_accuracy:.4f}")
torch.save(best_model_state_dict, os.path.join(model_dir, "model.pt"))
