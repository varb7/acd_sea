from setuptools import setup, find_packages

setup(
    name="acd_sea",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "pyyaml",
        "mlflow",
        # Add other dependencies from requirements.txt
    ],
)
