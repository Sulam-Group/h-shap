import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

data_dir = os.path.join("data")
explanation_dir = os.path.join("explanations")

exp_mapper = {
    "gradcam": r"Grad-CAM",
    "lime": r"LIME",
    "gradexp": r"GradientExp",
    "deepexp": r"DeepExp",
    "partexp/500": r"PartitionExp ($m = 500$)",
    "partexp/128": r"PartitionExp ($m = 128$)",
    "partexp/64": r"PartitionExp ($m = 64$)",
    "partexp/32": r"PartitionExp ($m = 32$)",
    "partexp/16": r"PartitionExp ($m = 16$)",
    "hexp/absolute_0": r"h-Shap ($\tau = 0$)",
    "hexp/relative_70": r"h-Shap ($\tau=70\%$)",
}

df_train = pd.read_json(os.path.join(data_dir, "training.json"))
df_test = pd.read_json(os.path.join(data_dir, "test_cropped.json"))
frame = [df_train, df_test]
gt_df = pd.concat(frame, ignore_index=True)
image_name = []
for _, row in gt_df.iterrows():
    image_name.append(os.path.basename(row["image"]["pathname"]).split(".")[0])
gt_df["image_name"] = image_name
gt_df.set_index("image_name", inplace=True)

true_positive = np.load(os.path.join(explanation_dir, "true_positive.npy"))

for exp_name, exp_title in exp_mapper.items():
    print(f"Processing {exp_name}")
    explainer_dir = os.path.join(explanation_dir, exp_name)

    df = []
    comp_time = np.load(os.path.join(explainer_dir, "comp_times.npy"))

    for i, image_path in enumerate(tqdm(true_positive)):
        image_name = os.path.basename(image_path).split(".")[0]
        ground_truth = np.zeros((1200, 1600), dtype=np.bool_)
        cell = gt_df.at[image_name, "objects"]
        for c in cell:
            category = c["category"]
            if category == "trophozoite":
                bbox = c["bounding_box"]
                ul_r = bbox["minimum"]["r"]
                ul_c = bbox["minimum"]["c"]
                br_r = bbox["maximum"]["r"]
                br_c = bbox["maximum"]["c"]
                ground_truth[ul_r : br_r + 1, ul_c : br_c + 1] = 1
        explanation = np.load(os.path.join(explainer_dir, f"{image_name}.npy"))

        eps = 1e-06
        ground_truth = ground_truth.flatten()
        explanation = (explanation > eps).flatten()

        score = f1_score(ground_truth, explanation)
        runtime = comp_time[i]
        if runtime > 0 and score > 0:
            df.append(
                {
                    "exp_name": exp_name,
                    "exp_title": exp_title,
                    "comp_time": runtime,
                    "score": score,
                }
            )
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(explainer_dir, f"f1_scores.csv"))
