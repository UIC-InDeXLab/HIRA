from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hira",
    version="0.1.0",
    description="Hierarchical Range-Searching Attention for Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hira",
    # packages=find_packages(),
    packages=["hira", "hira.attention", "hira.cache", "hira.index", "hira.search", "hira.utils", "hira.kernels", "hira.tests"],
    package_dir={"hira": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "faiss": [
            "faiss-cpu>=1.7.0",  # or faiss-gpu for GPU support
        ],
        "examples": [
            "python-dotenv>=0.19.0",
            "huggingface-hub>=0.10.0",
        ],
    },
)
