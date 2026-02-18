from setuptools import setup, find_packages

setup(
    name="pathcond",
    version="0.1.0",
    description="Path-conditioned training: a principled way to rescale ReLU neural networks",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Arthur Lebeurrier, Titouan Vayer, Rémi Gribonval",
    # TODO: add author_email
    url="[TODO: GitHub repo URL]",
    license="CC BY 4.0",

    packages=find_packages(exclude=["expes*", "figures*", "data*", "venv*"]),

    python_requires=">=3.9",

    install_requires=[
        "torch>=2.0.0",
    ],

    extras_require={
        # pip install -e ".[expes]"
        "expes": [
            "torchvision>=0.15.0",
            "mlflow>=2.0.0",
            "tqdm>=4.60.0",
            "seaborn>=0.12.0",
            "matplotlib>=3.5.0",
            "numpy>=1.23.0",
            "scipy>=1.9.0",
        ],
        # pip install -e ".[demo]"
        "demo": [
            "torchvision>=0.15.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "numpy>=1.23.0",
            "scikit-learn>=1.1.0",
        ],
        # pip install -e ".[all]"
        "all": [
            "torchvision>=0.15.0",
            "mlflow>=2.0.0",
            "tqdm>=4.60.0",
            "seaborn>=0.12.0",
            "matplotlib>=3.5.0",
            "numpy>=1.23.0",
            "scipy>=1.9.0",
            "jupyter>=1.0.0",
            "scikit-learn>=1.1.0",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",  # CC BY 4.0 has no PyPI trove classifier
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],

    keywords=[
        "deep learning",
        "neural networks",
        "optimization",
        "rescaling",
        "ReLU",
        "conditioning",
        "initialization",
    ],
)
