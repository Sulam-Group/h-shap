import os
import sys
from setuptools import setup, find_packages
from sys import platform

PACKAGE_NAME = "hshap"
AUTHOR = ("Jacopo Teneggi", "Alexandre Luster", "Jeremias Sulam")
AUTHOR_EMAIL = "jtenegg1@jhu.edu"

# Find mouselight_code version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "hshap", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]
        print(VERSION)

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
)
