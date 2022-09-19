import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm

data_dir = os.path.join("data")
ground_truth_dir = os.path.join(data_dir, "ground_truth")
explanation_dir = os.path.join("explanations")

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
        comp_times = np.load(os.path.join(explainer_dir, f"comp_times_{n}.npy"))

        for i, image_path in enumerate(tqdm(true_positives[n])):
            image_name = os.path.basename(image_path).split(".")[0]
            ground_truth = np.zeros((100, 120))
            positions = np.load(
                os.path.join(ground_truth_dir, str(n), f"{image_name}.npy")
            )
            for position in positions:
                top_left = position[0]
                bottom_right = position[1]
                for k in range(9):
                    ground_truth[top_left[1] + k, top_left[0] + k] = 1
                    ground_truth[bottom_right[1] - k, top_left[0] + k] = 1
            explanation = np.load(os.path.join(explainer_dir, f"{image_name}.npy"))

            eps = 1e-06
            ground_truth = ground_truth.flatten()
            explanation = (explanation > eps).flatten()

            score = f1_score(ground_truth.flatten(), explanation.flatten() > 0)
            runtime = comp_times[i]

            df.append(
                {
                    "exp_name": exp_name,
                    "exp_title": exp_title,
                    "n": n,
                    "comp_time": runtime,
                    "score": score,
                }
            )
        df = pd.DataFrame(df)
        df.to_csv(os.path.join(explainer_dir, f"f1_scores_{n}.csv"))
