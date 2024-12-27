import setuptools
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements=[req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


__version__ = '0.0.0'

REPO_NAME = "Student_Performance_Analyser"
AUTHOR_USERNAME = "Mayuresh999"
SRC_REPO = "Student_Performance_Analyser"
AUTHOR_EMAIL = "mayuresh.madiwale@yahoo.com"


setuptools.setup(
    name=SRC_REPO,
    version= __version__,
    author= AUTHOR_USERNAME,
    author_email= AUTHOR_EMAIL,
    description= 'This project is a practice project with ML and MLOPS tools.',
    long_description= long_description,
    long_description_content= "text/markdown",
    url=f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}",
    project_urls= {"Student_Performance_Analyser" : f"https://github.com/{AUTHOR_USERNAME}/{REPO_NAME}/issues"},
    package_dir= {"" : "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires = get_requirements('requirements.txt')
)