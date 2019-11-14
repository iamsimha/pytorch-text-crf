import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="pytorch-text-crf",
    version="0.1",
    description="A simple crf module written in pytorch. The implementation is based\
                https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/iamsimha/pytorch-text-crf",
    author="iamsimha",
    author_email="jayasimhatlr@gmail.com",
    license="Apache License, Version 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=['*__pycache__*', '*.tests.*', 'tests.*', 'tests']),
    include_package_data=True,
    install_requires=["torch"]
)
