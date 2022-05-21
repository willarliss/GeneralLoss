import os
from setuptools import setup

REQS = './requirements/requirements.txt'

with open(REQS, 'r') as infile:
    requirements = infile.read().split('\n')

if __name__ == '__main__':

    os.system('python3 -m pip install -U pip setuptools')
    os.system(f'pip3 install -r {REQS}')

    setup(
        name='Minimizers',
        packages=['minimizers'],
        install_requires=requirements,
        version='1.0.0',
        python='>=3.10.4',
        description="A general framework for minimizing custom loss functions using Scipy's and Scikit-Learn.",
        classifiers=['Programming Language :: Python :: 3'],
    )
