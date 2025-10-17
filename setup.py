"""
Script de configuración para Linksy - Predictor de Ventas IA
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="linksy-sales-predictor",
    version="1.0.0",
    author="Linksy Team",
    author_email="team@linksy.com",
    description="Sistema de predicción de ventas con IA que integra variables económicas, del mercado y del producto",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/linksy/sales-predictor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "linksy-demo=run_demo:main",
            "linksy-api=api:main",
            "linksy-app=streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords=[
        "sales prediction",
        "machine learning",
        "artificial intelligence",
        "economic variables",
        "market analysis",
        "demand forecasting",
        "business intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/linksy/sales-predictor/issues",
        "Source": "https://github.com/linksy/sales-predictor",
        "Documentation": "https://linksy.readthedocs.io/",
    },
)

