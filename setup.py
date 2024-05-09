from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='scientific_paper_classifier',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
      packages=find_packages(),
)
