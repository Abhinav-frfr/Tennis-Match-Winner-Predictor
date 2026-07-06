from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    '''
    This function helps me to return the list of 
    requirements mentioned in the requirements.txt file

    '''

    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="ml_project",
    version="0.1.0",
    packages=find_packages(),
    author="Abhinav",
    author_email="abhinav.fr807@gmail.com",
    install_requires=get_requirements("requirements.txt")
)

