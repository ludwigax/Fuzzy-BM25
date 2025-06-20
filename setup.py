from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fuzzy_bm25",
    version="0.1.0",
    author="Ludwig",
    author_email="yuzeliu@gmail.com",
    description="High-performance BM25 implementation with fuzzy search capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ludwigax/fuzzy_bm25",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "nltk>=3.6",
    ],
    keywords="bm25 information-retrieval fuzzy-search text-search nlp",
    project_urls={
        "Source": "https://github.com/ludwigax/fuzzy_bm25",
    },
)