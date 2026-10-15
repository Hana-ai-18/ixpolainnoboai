"""
TC-PhysXAI — setup.py
======================
Minimal setup for importing PhysXAI as a package.
On Kaggle: no install needed — just add the parent dir to sys.path.
"""

from setuptools import setup, find_packages

setup(
    name="TC-PhysXAI",
    version="1.0.0",
    author="Nguyen Hong Gam",
    author_email="nguyenhonggamuit@gmail.com",
    description=(
        "TC-PhysXAI: Physics-Grounded XAI for Probabilistic Tropical Cyclone Prediction "
        "via Conditional Flow Matching"
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "vis": ["seaborn>=0.12.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ],
)
