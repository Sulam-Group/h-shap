import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import hshap
import shap
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from pytorch_grad_cam import GradCAM
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join("data")
trophozoite_dir = os.path.join(data_dir, "trophozoite")
model_dir = os.path.join("pretrained_model")
explanation_dir = os.path.join("explanations")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(
    torch.load(os.path.join(model_dir, "model.pt"), map_location=device)
)
model = model.to(device)
model.eval()
# The first forward pass in benchmark mode will allocate large amounts of memory
# for testing purposes. This may alter the runtime readings of the first explanation.
# Hence, we dry run the model, and then clear the GPU memory.
# Reference: https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/4
x = torch.randn(1, 3, 1200, 1600, device=device)
model(x)
torch.cuda.empty_cache()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
)


def cam_init():
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=True)
    return cam


def cam_explain(cam, image_t):
    torch.set_grad_enabled(True)
    image_t = image_t.unsqueeze(0)
    t0 = time.time()
    explanation = cam(input_tensor=image_t)
    t = time.time()
    runtime = round(t - t0, 6)
    return explanation.squeeze(), runtime


def lime_init():
    limexp = lime_image.LimeImageExplainer()
    return limexp


def lime_explain(explainer, image_rgb):
    def f(x):
        x = torch.stack(tuple(transform(i) for i in x), dim=0)
        x = x.to(device)
        output = model(x)
        p = F.softmax(output, dim=1)
        return p.detach().cpu().numpy()

    image = np.array(image_rgb)
    t0 = time.time()
    explanation = explainer.explain_instance(
        image, f, top_labels=1, num_samples=100, segmentation_fn=segmentation_fn
    )
    _, explanation = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=len(explanation.segments),
        hide_rest=False,
    )
    t = time.time()
    runtime = round(t - t0, 6)
    return explanation, runtime


def gradexp_init():
    dataset = datasets.ImageFolder(os.path.join(trophozoite_dir, "train"), transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    gradexp = shap.GradientExplainer(model, X)
    return gradexp


def gradexp_explain(gradexp, image_t):
    torch.set_grad_enabled(True)
    image_t = image_t.unsqueeze(0)
    t0 = time.time()
    explanation, _ = gradexp.shap_values(image_t, ranked_outputs=1, nsamples=10)
    t = time.time()
    runtime = round(t - t0, 6)
    return explanation[0][0].sum(0), runtime


def deepexp_init():
    dataset = datasets.ImageFolder(os.path.join(trophozoite_dir, "train"), transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
    _iter = iter(dataloader)
    X, _ = next(_iter)
    X = X.to(device)
    deepexp = shap.DeepExplainer(model, X)
    return deepexp


def deepexp_explain(deepexp, image_t):
    image_t = image_t.unsqueeze(0)
    t0 = time.time()
    explanation, _ = deepexp.shap_values(image_t, ranked_outputs=1)
    t = time.time()
    runtime = round(t - t0, 6)
    return explanation[0][0].sum(0), runtime


def partexp_init():
    ref = torch.load(os.path.join(explanation_dir, "reference.pt"))
    ref = ref.permute(1, 2, 0)
    ref = ref.numpy()
    masker = shap.maskers.Image(ref)

    def f(x):
        x = torch.tensor(x).float()
        x = x.permute(0, 3, 1, 2)
        x = x.to(device)
        return model(x).detach().cpu().numpy()[..., -1]

    partexp = shap.Explainer(f, masker)
    return partexp


def partexp_explain(partexp, image_np):
    t0 = time.time()
    explanation = partexp(
        np.expand_dims(image_np, axis=0), max_evals=max_evals, fixed_context=0
    )
    t = time.time()
    runtime = round(t - t0, 6)
    return explanation.values[0].sum(axis=-1), runtime


def hexp_init():
    ref = torch.load(os.path.join(explanation_dir, "reference.pt"), map_location=device)
    s = 80
    hexp = hshap.src.Explainer(
        model=model,
        background=ref,
        s=s,
    )
    return hexp


def hexp_explain(hexp, image_t):
    torch.set_grad_enabled(False)
    t0 = time.time()
    explanation = hexp.explain(
        image_t,
        label=1,
        threshold_mode=threshold_mode,
        threshold=threshold_value,
        batch_size=2,
    )
    t = time.time()
    runtime = round(t - t0, 6)
    torch.cuda.empty_cache()
    return explanation.squeeze(), runtime


exp_mapper = [
    {"name": "gradcam", "init": cam_init, "explain": cam_explain},
    {"name": "gradexp", "init": gradexp_init, "explain": gradexp_explain},
    {"name": "deepexp", "init": deepexp_init, "explain": deepexp_explain},
    {"name": "partexp/500", "init": partexp_init, "explain": partexp_explain},
    {"name": "partexp/128", "init": partexp_init, "explain": partexp_explain},
    {"name": "partexp/64", "init": partexp_init, "explain": partexp_explain},
    {"name": "partexp/32", "init": partexp_init, "explain": partexp_explain},
    {"name": "partexp/16", "init": partexp_init, "explain": partexp_explain},
    {"name": "hexp/absolute_0", "init": hexp_init, "explain": hexp_explain},
    {"name": "hexp/relative_70", "init": hexp_init, "explain": hexp_explain},
    {"name": "lime", "init": lime_init, "explain": lime_explain},
]

true_positives = np.load(os.path.join(explanation_dir, "true_positive.npy"))

for exp in exp_mapper:
    exp_name = exp["name"]
    explainer_dir = os.path.join(explanation_dir, exp_name)
    os.makedirs(explainer_dir, exist_ok=True)
    explainer = exp["init"]()
    explain = exp["explain"]
    print("Initialized %s" % exp_name)

    comp_times = []
    for i, image_path in enumerate(true_positives):
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image_t = transform(image).to(device)
        image_np = image_t.permute(1, 2, 0).cpu().numpy()
        image_rgb = image.convert("RGB")
        if "gradcam" in exp_name:
            explanation, runtime = explain(explainer, image_t)
        elif "lime" in exp_name:
            segmentation_fn = SegmentationAlgorithm(
                "quickshift", kernel_size=4, max_dist=200, ratio=0.2
            )
            explanation, runtime = explain(explainer, image_rgb)
        elif "gradexp" in exp_name or "deepexp" in exp_name:
            explanation, runtime = explain(explainer, image_t)
        elif "partexp" in exp_name:
            max_evals = int(exp_name.split("/")[1])
            explanation, runtime = explain(explainer, image_np)
        elif "hexp" in exp_name:
            _threshold = exp_name.split("/")[1].split("_")
            threshold_mode = _threshold[0]
            threshold_value = int(_threshold[1])
            explanation, runtime = explain(explainer, image_t)
        else:
            raise NotImplementedError(f"{exp_name} is not implemented")
        comp_times.append(runtime)
        print(f"{exp_name}: {i+1}/{len(true_positives)} runtime={runtime:4f} s")
        np.save(os.path.join(explainer_dir, image_name), explanation)
    np.save(os.path.join(explainer_dir, f"comp_times.npy"), comp_times)
