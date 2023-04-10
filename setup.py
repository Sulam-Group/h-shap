from setuptools import setup, find_packages

VERSION = "0.1.6"
PACKAGE_NAME = "h-shap"
DESCRIPTION = "Fast Hierarchical Games for Image Explanations"
LONG_DESCRIPTION = "`h-shap` provides a fast, hierarchical implementation of Shapley coefficients for image explanations. It is exact, and it does not rely on approximation. In binary classification scenarios, `h-shap` guarantees an exponential computational advantage when explaining an important concept contained in the image."
AUTHOR = "Jacopo Teneggi"
AUTHOR_EMAIL = "jtenegg1@jhu.edu"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url="https://github.com/Sulam-Group/h-shap",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "torch"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
