from setuptools import setup, find_packages

setup(
    name="archid",
    version="0.9.0",
    description="A higher level deep learning framework built on Euclid",
    author="Dra3don",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "euclid-ml"
        ]
)