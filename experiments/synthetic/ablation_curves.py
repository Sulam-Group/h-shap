import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from model import Net
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
model_dir = os.path.join("pretrained_model")
explanation_dir = os.path.join("explanations")

torch.set_grad_enabled(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = Net()
model.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt")),
)
model = model.to(device)
model.eval()
x = torch.randn(1, 3, 100, 120, device=device)
model(x)
torch.cuda.empty_cache()

mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

ref = torch.load(os.path.join(explanation_dir, "reference.pt"), map_location=device)

A = ref.size(1) * ref.size(2)
L = 200
exp_perturbation_size = np.linspace(np.log10(1 / A), 0, L)
relative_perturbation_size = np.sort(10 ** (exp_perturbation_size))
perturbation_size = np.round(A * relative_perturbation_size)
perturbation_size = np.array(perturbation_size, dtype="int")
perturbation_size = np.unique(perturbation_size)

exp_mapper = {
    "gradcam": r"Grad-CAM",
    "lime": r"LIME",
    "gradexp": r"GradientExp",
    "deepexp": r"DeepExp",
    "partexp/500": r"PartitionExp ($m = 500$)",
    "partexp/64": r"PartitionExp ($m = 64$)",
    "partexp/32": r"PartitionExp ($m = 32$)",
    "partexp/16": r"PartitionExp ($m = 16$)",
    "hexp/absolute_0": r"h-Shap ($\tau = 0$)",
    "hexp/relative_70": r"h-Shap ($\tau=70\%$)",
}

c = [1, 6]
true_positives = np.load(
    os.path.join(explanation_dir, "true_positive.npy"), allow_pickle=True
).item()

for n in c:
    for exp_name, exp_title in exp_mapper.items():
        print(f"({n}) Processing {exp_name}")
        explainer_dir = os.path.join(explanation_dir, exp_name)

        df = []
        for i, image_path in enumerate(tqdm(true_positives[n])):
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
            image = transform(image).to(device)
            explanation = np.load(os.path.join(explainer_dir, f"{image_name}.npy"))

            activation_threshold = 0
            k, l = np.where(explanation > activation_threshold)
            scores = explanation[k, l]
            rank = np.argsort(scores)
            M = len(scores)

            _perturbation_size = perturbation_size[perturbation_size <= M]
            m = len(_perturbation_size)

            original_output = model(image.unsqueeze(0))
            original_logit = F.softmax(original_output, dim=1)[:, 1]

            r = 16
            for j in range(0, m, r):
                batch_size = _perturbation_size[j : j + r]
                batch = image.repeat(len(batch_size), 1, 1, 1)
                for u, s in enumerate(batch_size):
                    idx = rank[-s:]
                    pk, pl = k[idx], l[idx]
                    batch[u, :, pk, pl] = ref[:, pk, pl]
                output = model(batch)
                logit = F.softmax(output, dim=1)[:, 1]
                for s, p in zip(batch_size, logit):
                    df.append(
                        {
                            "index": f"{exp_name}_{i}_{s}",
                            "exp_name": exp_name,
                            "exp_title": exp_title,
                            "size": s / A,
                            "logit": p.item() / original_logit.item(),
                        }
                    )
        df = pd.DataFrame(df)
        df.set_index("index", inplace=True)
        df.to_csv(os.path.join(explainer_dir, f"ablation_curves_{n}.csv"))
