from setuptools import find_packages, setup

setup(
    name="fp_quant",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Andrei Panferov",
    author_email="andrei@panferov.org",
    description="A Python library for FP-Quant",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IST-DASLab/FP-Quant",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.7.0",
        "scipy>=1.13.0",
        "triton>=3.3.0",
    ],
)
