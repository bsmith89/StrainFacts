import setuptools

# Long description and versioning code copied with modifications from
# <https://github.com/blackjax-devs/blackjax/blob/main/setup.py>

with open("README.md") as f:
    long_description = f.read()


def get_version(path):
    """Get the package's version number.
    We fetch the version  number from the `__version__` variable located in the
    package root's `__init__.py` file. This way there is only a single source
    of truth for the package's version number.
    """
    with open(path) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="StrainFacts",
    version=get_version("sfacts/__init__.py"),
    long_description=long_description,
    description="Factorize metagenotypes to infer strains and their abundances",
    url="http://github.com/bsmith89/StrainFacts",
    author="Byron J. Smith",
    author_email="me@byronjsmith.com",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
        "pyro-ppl",
        "xarray",
        "netcdf4",
        "pandas",
        "matplotlib",
        "numpy",
        "scipy",
        "tqdm",
        "scikit-learn",
        "seaborn",
    ],
    dependency_links=[],
    entry_points={"console_scripts": ["sfacts = sfacts.__main__:main"]},
    zip_safe=False,
)
