from pathlib import Path

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> list[str]:
    requirements_path = Path(file_path)
    return [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="retention_ai",
    version="0.1.0",
    author="Ajit",
    author_email="ajitpshetti@gmail.com",
    description="Churn prediction API and training pipeline for customer retention.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
)
