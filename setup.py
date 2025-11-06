from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="powergrid",
    version="0.0.2",
    description="Power Grid Gymnasium environment built on pandapower.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hepengli/powergrid",
    author="Hepeng Li",
    author_email="hepeng.li@maine.edu",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(),
    # packages=find_packages(where="powergrid", exclude=("tests", "docs")),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "gymnasium",                          # <— fixed spelling
        "pandapower",
        "numpy",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest",
            "ruff",
            "black",
            "mypy",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    project_urls={
        "Source": "https://github.com/hepengli/powergrid",
        "Issues": "https://github.com/hepengli/powergrid/issues",
    },
)
