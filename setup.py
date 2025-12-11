from setuptools import setup, find_packages

setup(
    name="pathcond",
    version="0.1.0",
    description="Minimal MLP (ReLU) for MNIST",
    author="Arthur Lebeurrier",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "torchvision>=0.16",
        "numpy>=1.26",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "contextlib2>=0.6.0",
        "pathlib-mate>=1.0.1",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "dev": ["pytest>=8"],
    },
    entry_points={
        "console_scripts": [
            "mnist-train = expes.cli:main",
            "compare-g-diag-g = plot_diag_vs_full:main",
            "mnist-plot-curves = expes.cli_plot:main",
            "moons-multi-lr = expes.cli_train_multi_lr_moons:main",
            "moons-boxplots = expes.cli_boxplots_moons:main",
            "ts-multi-lr = expes.cli_train_multi_lr_teacher_student:main",
            "ts-boxplots = expes.cli_boxplots_teacher_student:main",
            "resnet-mnist-train = expes.cli_resnet:main",
            "resnet-mnist-plot-curves = expes.cli_plot_resnet:main",
            "plot-mnist-resnet = expes.cli_plot_resnet:main",
        ],
    },
)
