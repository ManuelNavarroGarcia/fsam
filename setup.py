import pathlib

from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="fsam",
    license="MIT",
    version="0.1.0",
    packages=find_packages(),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Manuel Navarro García",
    author_email="manuelnavarrogithub@gmail.com",
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    url="https://github.com/ManuelNavarroGarcia/fsam/",
    download_url="https://github.com/ManuelNavarroGarcia/fsam/archive/refs/tags/0.1.0.tar.gz",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "cpsplines",
        "typer",
        "gurobipy >= 10.0.0",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    extras_require={"dev": ["black", "ipykernel", "pip-tools", "pytest", "ipywidgets"]},
)
