import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

data_dir = os.path.join("data")
annotation_dir = os.path.join(data_dir, "Annotations", "Annotations")
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

gt_df = []
for sequence_name in ["daySequence1", "daySequence2"]:
    _df = pd.read_csv(
        os.path.join(annotation_dir, sequence_name, "frameAnnotationsBOX.csv"), sep=";"
    )
    gt_df.append(_df)
gt_df = pd.concat(gt_df, axis=0)
gt_df["ImageName"] = gt_df.apply(
    lambda row: os.path.basename(row["Filename"]).split(".")[0], axis=1
)
gt_df["Label"] = gt_df.apply(
    lambda row: 1 if "go" in row["Annotation tag"] else 0, axis=1
)
gt_df.set_index("ImageName", inplace=True)

true_positive = np.load(os.path.join(explanation_dir, "true_positive_explained.npy"))

for exp_name, exp_title in exp_mapper.items():
    print(f"Processing {exp_name}")
    explainer_dir = os.path.join(explanation_dir, exp_name)

    df = []
    comp_time = np.load(os.path.join(explainer_dir, "comp_times.npy"))

    for i, image_path in enumerate(tqdm(true_positive)):
        image_name = os.path.basename(image_path).split(".")[0]
        ground_truth = np.zeros((960, 1280), dtype=np.bool_)
        annotations = gt_df.loc[[image_name]]
        for _, annotation in annotations.iterrows():
            if annotation["Label"] == 1:
                upper_left_c = int(annotation["Upper left corner X"])
                upper_left_r = int(annotation["Upper left corner Y"])
                lower_right_c = int(annotation["Lower right corner X"])
                lower_right_r = int(annotation["Lower right corner Y"])
                ground_truth[
                    upper_left_r : lower_right_r + 1, upper_left_c : lower_right_c + 1
                ] = 1

        explanation = np.load(os.path.join(explainer_dir, f"{image_name}.npy"))

        eps = 1e-06
        ground_truth = ground_truth.flatten()
        explanation = (explanation > eps).flatten()

        score = f1_score(ground_truth, explanation)
        runtime = comp_time[i]

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
