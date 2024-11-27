from setuptools import setup, find_packages
from typing import List

HYPHEN_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('/n', ' ') for req in requirements]

        if HYPHEN_DOT in requirements:
            requirements.remove(HYPHEN_DOT)
    return requirements 


setup(
    name = 'mlproject',
    version = '0.0.1',
    author='Shiv Dixit',
    author_email= 'shivdixit61@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'), # add any additional packages that
)