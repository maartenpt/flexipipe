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
from setuptools.command.install import install

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
    "requests>=2.31.0",
    "tabulate>=0.9.0",
]

EXTRAS = {
    "fasttext": ["fasttext-numpy2>=0.9.2.post2"],
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
        # Try to build flexitag_py extension after building other extensions
        self._build_flexitag_py()

    def _build_flexitag_py(self):
        """Build flexitag_py C++ extension using CMake if possible."""
        import os
        import subprocess
        
        # Check if we're in non-interactive mode (TEITOK installations)
        noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
        quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in ("1", "true", "yes")
        is_teitok = noninteractive or quiet_install
        
        # Check if flexitag directory exists
        flexitag_dir = DEPS_BASE / "flexitag"
        if not flexitag_dir.exists() or not (flexitag_dir / "CMakeLists.txt").exists():
            if not is_teitok:
                print("[flexipipe] flexitag directory not found, skipping flexitag_py build")
            return
        
        # Check if CMake is available
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            if not is_teitok:
                print("[flexipipe] CMake not found, skipping flexitag_py build (Python fallback will be used)")
            return
        
        # Build flexitag_py
        build_dir = flexitag_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        try:
            if not is_teitok:
                print("[flexipipe] Building flexitag_py C++ extension...")
            
            # Configure with CMake
            cmake_cmd = [
                "cmake", "..",
                "-DFLEXITAG_BUILD_PYTHON=ON",
            ]
            result = subprocess.run(
                cmake_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if not is_teitok:
                    print(f"[flexipipe] CMake configuration failed: {result.stderr}")
                    print("[flexipipe] Continuing without flexitag_py (Python fallback will be used)")
                return
            
            # Build flexitag_py target
            build_cmd = ["cmake", "--build", ".", "--target", "flexitag_py", "-j"]
            result = subprocess.run(
                build_cmd,
                cwd=build_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                if not is_teitok:
                    print(f"[flexipipe] flexitag_py build failed: {result.stderr}")
                    print("[flexipipe] Continuing without flexitag_py (Python fallback will be used)")
                return
            
            # Check if the module was built
            so_files = list(build_dir.glob("flexitag_py*.so"))
            if not so_files:
                # Try .pyd for Windows
                so_files = list(build_dir.glob("flexitag_py*.pyd"))
            
            if so_files:
                built_module = so_files[0]
                if not is_teitok:
                    print(f"[flexipipe] Successfully built flexitag_py: {built_module}")
                
                # Copy the built module to the flexipipe package directory so it can be imported
                # This ensures it works even when flexitag/build is not in the expected location
                try:
                    # During build, copy to build directory so it gets included in the wheel/installation
                    if hasattr(self, 'build_lib') and self.build_lib:
                        # We're in build mode - copy to build directory
                        target_dir = Path(self.build_lib) / "flexipipe"
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / built_module.name
                        shutil.copy2(built_module, target_path)
                        if not is_teitok:
                            print(f"[flexipipe] Copied flexitag_py to {target_path}")
                        
                        # Also ensure flexitag/build directory structure is preserved in the build
                        # This allows the module to be found via sys.path manipulation in engine.py/teitok.py
                        flexitag_target = Path(self.build_lib) / "flexitag" / "build"
                        flexitag_target.parent.mkdir(parents=True, exist_ok=True)
                        if not flexitag_target.exists():
                            flexitag_target.mkdir(parents=True, exist_ok=True)
                        # Copy the built module to flexitag/build as well
                        flexitag_module_path = flexitag_target / built_module.name
                        shutil.copy2(built_module, flexitag_module_path)
                        
                        # Also copy the source flexitag directory structure if it exists
                        # This ensures the build directory is available after installation
                        if flexitag_dir.exists():
                            # Copy flexitag source to build_lib (but skip build directory to avoid recursion)
                            import distutils.dir_util
                            flexitag_build_target = Path(self.build_lib) / "flexitag"
                            if not flexitag_build_target.exists():
                                # Copy flexitag directory but exclude the build subdirectory
                                distutils.dir_util.copy_tree(
                                    str(flexitag_dir),
                                    str(flexitag_build_target),
                                    preserve_mode=0,
                                    preserve_times=0,
                                    update=0,
                                    verbose=0,
                                )
                                # Remove the original build directory if it was copied
                                copied_build = flexitag_build_target / "build"
                                if copied_build.exists():
                                    import shutil
                                    shutil.rmtree(copied_build)
                            # Ensure the build directory exists with the module
                            flexitag_build_target.mkdir(parents=True, exist_ok=True)
                            flexitag_build_dir = flexitag_build_target / "build"
                            flexitag_build_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(built_module, flexitag_build_dir / built_module.name)
                    else:
                        # Try to find where flexipipe will be installed
                        # This is a fallback - the module will still be found via sys.path in engine.py
                        pass
                except Exception as copy_error:
                    # If copying fails, the module will still be found via sys.path manipulation
                    if not is_teitok:
                        print(f"[flexipipe] Note: Could not copy flexitag_py to package directory: {copy_error}")
            else:
                if not is_teitok:
                    print("[flexipipe] flexitag_py module not found after build")
        except Exception as e:
            if not is_teitok:
                print(f"[flexipipe] Error building flexitag_py: {e}")
                print("[flexipipe] Continuing without flexitag_py (Python fallback will be used)")
            # Don't fail the installation if flexitag_py build fails

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


def install_wrapper_script():
    """Interactive installation of the flexipipe wrapper script."""
    import os
    import shutil
    import stat
    import subprocess
    
    # Check for non-interactive mode (for automated installs like from PHP)
    noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
    quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in ("1", "true", "yes")
    
    if noninteractive or quiet_install:
        # Skip wrapper script installation in non-interactive mode
        return
    
    print("\n" + "="*70)
    print("Flexipipe Wrapper Script Installation")
    print("="*70)
    print("\nThis will install a wrapper script that allows you to run")
    print("'flexipipe' directly instead of 'python -m flexipipe'.")
    print()
    
    # Ask about virtual environment
    use_venv = input("Do you want to use a virtual environment for flexipipe? [y/N]: ").strip().lower()
    venv_path = None
    if use_venv in ('y', 'yes'):
        venv_path = input("Enter the path to your virtual environment (or press Enter to skip): ").strip()
        if not venv_path:
            venv_path = None
        elif not Path(venv_path).exists():
            print(f"Warning: Virtual environment path does not exist: {venv_path}")
            use_venv = input("Continue anyway? [y/N]: ").strip().lower()
            if use_venv not in ('y', 'yes'):
                venv_path = None
    
    # Ask where to install the script
    print("\nWhere would you like to install the wrapper script?")
    print("  1. /usr/local/bin (system-wide, requires sudo)")
    print("  2. ~/bin (user-local, add to PATH)")
    print("  3. Custom location")
    print("  4. Skip installation")
    
    choice = input("Enter choice [1-4] (default: 4): ").strip() or "4"
    
    if choice == "4":
        print("Skipping wrapper script installation.")
        print("You can install it manually later by copying scripts/flexipipe to your PATH.")
        return
    
    # Determine installation path
    script_source = Path(__file__).parent / "scripts" / "flexipipe"
    if not script_source.exists():
        print(f"Error: Wrapper script not found at {script_source}")
        return
    
    if choice == "1":
        install_path = Path("/usr/local/bin/flexipipe")
        use_sudo = True
    elif choice == "2":
        install_path = Path.home() / "bin" / "flexipipe"
        install_path.parent.mkdir(parents=True, exist_ok=True)
        use_sudo = False
    elif choice == "3":
        custom_path = input("Enter installation path: ").strip()
        if not custom_path:
            print("No path provided, skipping installation.")
            return
        install_path = Path(custom_path).expanduser().resolve()
        install_path.parent.mkdir(parents=True, exist_ok=True)
        use_sudo = False
    else:
        print("Invalid choice, skipping installation.")
        return
    
    # Read the script and customize it
    script_content = script_source.read_text()
    
    # Add configuration at the top if venv_path is set
    if venv_path:
        venv_line = f'VENV_PATH="{venv_path}"'
        # Insert after the configuration comments
        lines = script_content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.startswith('# FLEXIPIPE_REPO_PATH') or line.startswith('# Optional: Set path'):
                insert_pos = i
                break
        lines.insert(insert_pos, venv_line)
        script_content = '\n'.join(lines)
    
    # Write to installation location
    try:
        if use_sudo:
            # Write to temp file first, then copy with sudo
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as tmp:
                tmp.write(script_content)
                tmp_path = tmp.name
            result = subprocess.run(
                ['sudo', 'cp', tmp_path, str(install_path)],
                capture_output=True,
                text=True
            )
            Path(tmp_path).unlink()
            if result.returncode != 0:
                print(f"Error installing script: {result.stderr}")
                return
            subprocess.run(['sudo', 'chmod', '+x', str(install_path)])
        else:
            install_path.write_text(script_content)
            install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
        
        print(f"\n✓ Wrapper script installed to: {install_path}")
        if choice == "2":
            print(f"\nNote: Make sure ~/bin is in your PATH.")
            print("Add this to your ~/.bashrc or ~/.zshrc:")
            print("  export PATH=\"$HOME/bin:$PATH\"")
        print()
    except Exception as e:
        print(f"Error installing wrapper script: {e}")


class FlexiInstall(install):
    """Custom install command that prompts for wrapper script installation."""
    
    def run(self):
        # Run the standard install
        install.run(self)
        
        # After installation, offer to install wrapper script
        # Only prompt if running interactively (not in automated builds)
        # Skip if FLEXIPIPE_NONINTERACTIVE or FLEXIPIPE_QUIET_INSTALL is set
        import os
        noninteractive = os.environ.get("FLEXIPIPE_NONINTERACTIVE", "").lower() in ("1", "true", "yes")
        quiet_install = os.environ.get("FLEXIPIPE_QUIET_INSTALL", "").lower() in ("1", "true", "yes")
        
        if sys.stdin.isatty() and not noninteractive and not quiet_install:
            try:
                install_wrapper_script()
            except KeyboardInterrupt:
                print("\n\nWrapper script installation cancelled.")
            except Exception as e:
                print(f"\nError during wrapper script installation: {e}")
                print("You can install it manually later.")


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
    package_data={
        "flexipipe": ["flexitag_py*.so", "flexitag_py*.pyd"],  # Include built flexitag_py module
        "": ["flexitag/build/flexitag_py*.so", "flexitag/build/flexitag_py*.pyd"],  # Include from flexitag/build
    },
    include_package_data=True,
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
    cmdclass={
        "build_ext": FlexiBuildExt,
        "install": FlexiInstall,
    },
    zip_safe=False,
)

