from os import path

from setuptools import setup, find_packages

version = "2023.2.0b1"


def read_requirements(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gempy_engine",
    version=version,
    author="Miguel de la Varga",
    author_email="miguel@terranigma-solutions.com",
    description="A Python package for GemPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gempy-project/gempy_engine",
    packages=find_packages(),
    license='EUPL-1.2',
    classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements/requirements.txt"),
    extras_require={
            "dev"   : read_requirements("requirements/dev-requirements.txt"),
            "opt"   : read_requirements("requirements/optional-requirements.txt"),
            "server": read_requirements("requirements/server-requirements.txt")
    },
    use_scm_version={
            "root"            : ".",
            "relative_to"     : __file__,
            "write_to"        : path.join("gempy_engine", "_version.py"),
            "fallback_version": "3.0.0"
    },
)
