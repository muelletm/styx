
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="stupid-bomelino", # Replace with your own username
    version="0.0.10",
    author="tino bomelino",
    author_email="tino@bomelino.de",
    description="AI helpers aber schön kurz",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bomelino/stupid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #install_requires = ["tfds-nightly"],
    python_requires='>=3.6',
)