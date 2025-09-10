from setuptools import setup, find_packages

setup(
    name="EasyTeacache",
    version="0.1.0",
    package_dir={"": "."},
    packages=find_packages(where="."),
    include_package_data=True,
)