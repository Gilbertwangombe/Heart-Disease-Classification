from setuptools import setup, find_packages
from typing import List  # Importing List from typing module

HYPHEN_E_DOT='-e .'
def get_requirements(file_path: str) -> List[str]:  # Added type hinting for file_path and return type List[str]
    '''
    this function will return a list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
            requirements=file_obj.readlines()
            requirements=[req.replace("\n","") for req in requirements]
            
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
                
            return requirements
    
setup(
    
    name='heart-disese-classification',
    version='0.0.1',
    description='A sample Python package',
    author='Gilbert',
    author_email='wangombegilbert1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)