#!/usr/bin/env python3
"""
Setup script for FlexiPipe
"""

import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

try:
    import pybind11
except ImportError as exc:
    raise RuntimeError(
        "pybind11 is required to build flexipipe. "
        "Ensure you're using pip>=21 and that pyproject.toml build requirements are respected, "
        "or install pybind11 manually (pip install pybind11)."
    ) from exc
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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

DEPS_BASE = Path(__file__).parent


def _download_and_extract(url: str, target_dir: Path) -> Path:
    if target_dir.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        archive_path = tmp_path / url.split("/")[-1]
        with urllib.request.urlopen(url) as response:
            data = response.read()
        archive_path.write_bytes(data)
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(tmp_path)
        extracted_roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if not extracted_roots:
            raise RuntimeError(f"Failed to extract dependency from {url}")
        root = extracted_roots[0]
        # Copy contents into target_dir
        for child in root.iterdir():
            dest = target_dir / child.name
            if dest.exists():
                continue
            if child.is_dir():
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    return target_dir


class FlexiBuildExt(build_ext):
    RAPIDJSON_URL = "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz"
    PUGIXML_URL = "https://github.com/zeux/pugixml/archive/refs/tags/v1.14.tar.gz"

    def run(self):
        rapidjson_include, pugixml_src_dir, pugixml_source = self._prepare_third_party()
        for ext in self.extensions:
            if ext.name == "flexipipe.pipeline_cpp":
                include_dirs = list(ext.include_dirs or [])
                include_dirs.extend([rapidjson_include, pugixml_src_dir])
                ext.include_dirs = include_dirs
                sources = list(ext.sources or [])
                if pugixml_source not in sources:
                    sources.append(pugixml_source)
                ext.sources = sources
        super().run()

    def _prepare_third_party(self):
        build_temp = Path(self.build_temp or "build")
        deps_dir = build_temp / "_deps"
        deps_dir.mkdir(parents=True, exist_ok=True)

        rapidjson_dir = _download_and_extract(self.RAPIDJSON_URL, deps_dir / "rapidjson")
        pugixml_dir = _download_and_extract(self.PUGIXML_URL, deps_dir / "pugixml")

        rapidjson_include = str((rapidjson_dir / "include").resolve())
        pugixml_src_dir = str((pugixml_dir / "src").resolve())
        pugixml_source = str((pugixml_dir / "src" / "pugixml.cpp").resolve())
        return rapidjson_include, pugixml_src_dir, pugixml_source


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
        ),
    ],
    cmdclass={"build_ext": FlexiBuildExt},
    zip_safe=False,
)

