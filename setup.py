from setuptools import find_packages,setup

from typing import List


HYPHEN_E_DOT='-e .'
def get_requirement(file_path:str)->List[str]:
    '''
    it will return the list of requirements
    '''
    requrement=[]
    with open(file_path) as file_obj:
        requrement=file_obj.readlines()
        requrement=[i.replace('\n',"") for i in requrement]

        if HYPHEN_E_DOT in requrement:
            requrement.remove(HYPHEN_E_DOT)
        return requrement

setup(
name='MobilePrice',
version='0.0.1',
author='Kunal',
author_email='kunalsamanta11@yahoo.in',
packages=find_packages(),
install_requires=get_requirement('requirement.txt')

)