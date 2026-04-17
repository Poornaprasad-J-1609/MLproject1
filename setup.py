from setuptools import find_packages,setup
from typing import List
HYPEN_DOT_E = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''this function will return the list of requirements'''
    requiremnts = []
    with open(file_path) as file_obj:
        requiremnts=file_obj.readlines()
        requiremnts=[req.replace("\n","")for req in requiremnts ]
   
        if HYPEN_DOT_E in requiremnts:
            requiremnts.remove(HYPEN_DOT_E)
    return requiremnts
setup(
    name='mlproject',
    version = '0.0.1',
    author="Prasad",
    author_email="1609prapus2711@gamil.com",
    packages=find_packages(),
    install_requires=['pandas','numpy','seaborn','matplotlib','scikit-learn']
)