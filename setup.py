"""
Minimal setup.py for the kaggle-mas package.

Install in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="kaggle-mas",
    version="0.1.0",
    description="Multi-Agent System for Kaggle rental property regression (mws-ai-agents-2026)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/YOUR_USERNAME/kaggle-mas",
    license="MIT",
    packages=find_packages(
        exclude=["tests*", "notebooks*", "docs*", "data*", "outputs*", "logs*"]
    ),
    python_requires=">=3.10",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "kaggle-mas=main:main",
        ],
    },
)
