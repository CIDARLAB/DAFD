import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DAFD",
    version="1.0.0",
    author="Ali Lashkaripour/David McIntyre",
    author_email="dpmc@bu.edu",
    description="Design automation tool for droplet generation'",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CIDARLAB/DAFD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)