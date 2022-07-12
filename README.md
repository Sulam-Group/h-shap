# h-Shap

[![codecov](https://codecov.io/gh/Sulam-Group/h-shap/branch/circleci-setup/graph/badge.svg?token=BTDZGRL8FK)](https://codecov.io/gh/Sulam-Group/h-shap)
[![circleci](https://circleci.com/gh/Sulam-Group/h-shap.svg?style=shield&circle-token=6570e24862d00e6ab61a24ffc93b4317fc50f262)](https://circleci.com/gh/Sulam-Group/h-shap)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.5914342.svg)](https://zenodo.org/record/5914342)

h-Shap provides a fast, hierarchical implementation of Shapley coefficients for image explanations. It is exact, and it does not rely on approximation. In binary classification scenarios, h-Shap guarantees an exponential computational advantage when explaining an important concept contained in the image (e.g. a sick cell in a blood smear, see example image below).

![Explanation example](./demo/explanations/2f6224be-50d0-4e85-94ef-88315df561b6.jpg)

## Installation

h-Shap is implemented in Python and it can be installed by cloning this repository.

```python
git clone https://github.com/Sulam-Group/h-shap.git
cd h-shap
pip install -e .
```

## Usage

h-Shap currently explains PyTorch models only. Given a model `model`, a reference input `ref`, and an input image `image`, run the following to initialize the explainer and compute the saliency map.

```python
# Initialize the explainer
hexp = hshap.src.Explainer(
  model=model, 
  background=ref, 
  min_size=s,
)
# Explain a prediction
explanation = hexp.explain(
  image,
  label=1,
  threshold_mode=threshold_mode,
  threshold=threshold,
)
```

where `s` is a minimal features size (e.g. `40 x 40` pixels), `threshold_mode` can be `"absolute"` or `"relative"`, and `threshold` is a relevance tolerance. See [`demo/`](https://github.com/Sulam-Group/h-shap/tree/master/demo) for further details on the parameters.

## Demo

[`demo/`](https://github.com/Sulam-Group/h-shap/tree/master/demo) contains a simple notebook to showcase h-Shap's functionality on the [BBBC041](https://bbbc.broadinstitute.org/BBBC041) dataset. The dataset comprises blood smears for malaria patients, and the model is trained to label positively all images that contain at least one _trophozoite_, one of the types of cells that indicate malaria. h-Shap then explains the model predictions and retrieves the sick cells in the images.

## Presentations

h-Shap received a Best Paper Award at the ICML 2021 Workshop on Interpretable Machine Learning in Healthcare ([IMLH21](https://sites.google.com/view/imlh2021/home)). Here is a link to our oral presentation: [video](https://drive.google.com/file/d/1j0T6uNresC3NAb7HnXv_3UyrLgbKNZd9/view?usp=sharing).

## Publications

When using h-Shap, please refer to our publication in 

```text
@article{Teneggi2022fast,
  title={Fast Hierarchical Games for Image Explanations}, 
  author={Teneggi, Jacopo and Luster, Alexandre and Sulam, Jeremias},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2022},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TPAMI.2022.3189849}}
```
