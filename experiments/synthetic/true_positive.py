import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")
explanation_dir = os.path.join("explanations")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = Net()
model.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
)
model.to(device)
model.eval()
torch.set_grad_enabled(False)

classes = range(1, 10)
true_positive = {c: [] for c in classes}
false_negative = []

batch_size = 64
t = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)
dataset = ImageFolder(os.path.join(data_dir, "test"), t)
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
            image_path = os.path.join(data_dir, "test", str(label.item()), image_name)
            if prediction[j] == 1:
                true_positive[label.item()].append(image_path)
            else:
                false_negative.append(image_path)

print(f"True positive count: {sum([len(u) for u in true_positive.values()])}")
print(f"False negative count: {len(false_negative)}")
np.save(
    os.path.join(explanation_dir, "true_positive"), true_positive, allow_pickle=True
)
