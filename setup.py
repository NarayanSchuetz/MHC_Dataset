from setuptools import setup, find_namespace_packages

setup(
    name="mhc_dataset",
    version="0.1",
    packages=find_namespace_packages(where="src", exclude=["venv", "venv.*"]),
    package_dir={'': 'src'}
)