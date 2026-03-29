from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).resolve().parent

setup(
    author="luniphys",
    name="number-neuralnetwork",
    version="1.0.0",
    description="A simple neural network to recognize handwritten digits.",
    long_description=(BASE_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "matplotlib",
        "PyQt6",
        "PyQt6-Charts",
    ],
    python_requires=">=3.8"
)