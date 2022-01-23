import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")
explanation_dir = os.path.join("explanations")
os.makedirs(explanation_dir, exist_ok=True)

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)
model.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
)
model.eval()
torch.set_grad_enabled(False)

true_positive = []
false_negative = []

batch_size = 4
t = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
dataset = ImageFolder(os.path.join(data_dir, "val"), t)
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
image_names = [os.path.basename(x[0]) for x in dataset.samples]

for i, data in enumerate(tqdm(dataloader)):
    input, label = data

    input = input.to(device)
    label = label.to(device)

    output = model(input)
    prediction = output.argmax(dim=1)

    for j, label in enumerate(label):
        if label > 0:
            image_id = i * batch_size + j
            image_name = image_names[image_id]
            image_path = os.path.join(data_dir, "val", str(label.item()), image_name)
            if prediction[j] == 1:
                true_positive.append(image_path)
            else:
                false_negative.append(image_path)

print(f"True positive count: {len(true_positive)}")
print(f"False negative count: {len(false_negative)}")
np.save(os.path.join(explanation_dir, "true_positive"), true_positive)

# Randomly select 300 images from the true positive list
true_positive = np.random.choice(true_positive, size=300, replace=False)
np.save(os.path.join(explanation_dir, "true_positive_explained.npy"), true_positive)
