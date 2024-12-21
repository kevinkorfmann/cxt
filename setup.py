from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cxt",  
    version="0.1",  
    author="Kevin Korfmann",
    author_email="kevin.korfmann@gmail.com",
    description="Coalescence x Translation",
    long_description=long_description,
    long_description_content_type="text/markdown",  
    url="https://github.com/kevinkorfmann/cxt",  
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
    ],
    python_requires=">=3.7",  # Specify the minimum Python version
    install_requires=[
        "numpy",
        "torch",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],  
        "docs": ["sphinx", "sphinx-rtd-theme"], 
    },
    include_package_data=True,  # Include files from MANIFEST.in
)
