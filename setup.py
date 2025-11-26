#!/usr/bin/env python3
"""
Setup script for FlexiPipe
"""

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import pybind11
import sys

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding='utf-8')

# Read version from flexipipe/__init__.py
version = "1.0.0"
init_file = Path(__file__).parent / "flexipipe" / "__init__.py"
if init_file.exists():
    for line in init_file.read_text(encoding='utf-8').split('\n'):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

BASE_REQUIREMENTS = [
    "pybind11>=2.10",
    "langcodes>=3.3.0",
    "language-data>=1.1.0",
    "pycountry>=23.12.0",
    "fasttext>=0.9.2",
    "requests>=2.31.0",
    "tabulate>=0.9.0",
]

EXTRAS = {
    "spacy": ["spacy>=3.7.0"],
    "stanza": ["stanza>=1.8.0"],
    "classla": ["classla>=2.1.0"],
    "flair": ["flair>=0.13.0", "torch>=2.6.0"],
    "transformers": [
        "torch>=2.6.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "accelerate>=0.20.0",
    ],
    "nametag": ["requests>=2.31.0"],
    "udpipe": ["requests>=2.31.0"],
    "udmorph": ["requests>=2.31.0"],
}

all_extras = sorted({dep for deps in EXTRAS.values() for dep in deps})
EXTRAS["all"] = all_extras
EXTRAS["dev"] = sorted(
    set(
        all_extras
        + [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    )
)

setup(
    name="flexipipe",
    version=version,
    description="Flexible transformer-based NLP pipeline for tagging, parsing, and normalization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/flexipipe",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=BASE_REQUIREMENTS,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "flexipipe=flexipipe.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, universal-dependencies, tagging, parsing, bert, transformers, normalization",
    ext_modules=[
        Extension(
            "flexipipe.viterbi_cpp",
            [
                "src/viterbi_cpp.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++17",
                "-O3",  # Optimize for speed
                "-Wall",
            ] if sys.platform != "win32" else [
                "/std:c++17",
                "/O2",  # Optimize for speed on Windows
            ],
        ),
        Extension(
            "flexipipe.pipeline_cpp",
            [
                "src/pipeline_pybind.cpp",
                "src/vocab_loader.cpp",
                "src/tokenizer.cpp",
                "src/normalizer.cpp",
                "src/contractions.cpp",
                "src/viterbi_optimized.cpp",
                "src/io_conllu.cpp",
                "src/io_teitok.cpp",
            ],
            include_dirs=[
                pybind11.get_include(),
                "src",
                "third_party/rapidjson/include",
                "third_party/pugixml/src",
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++17",
                "-O3",
                "-Wall",
            ] if sys.platform != "win32" else [
                "/std:c++17",
                "/O2",
            ],
            extra_objects=[
                "third_party/pugixml/src/pugixml.cpp",
            ] if sys.platform != "win32" else [],
        ),
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

