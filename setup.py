from setuptools import setup, find_packages


def read_requirements(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gempy_engine",
    version="0.1.0",
    author="Miguel de la Varga",
    author_email="miguel@terranigma-solutions.com",
    description="A Python package for GemPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terranigma-solutions/gempy_engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "opt": read_requirements("optional-requirements.txt"),
        "server": read_requirements("server-requirements.txt")
    }
)
