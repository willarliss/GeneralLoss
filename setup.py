from setuptools import setup

with open('requirements.txt', 'r') as infile:
    requirements = infile.read().split('\n')

if __name__ == '__main__':

    setup(
        name='Minimizers',
        packages=['minimizers'],
        install_requires=requirements,
        version='1.0.0',
        python='>=3.10.4',
        description="A general framework for minimizing custom loss functions using Scipy's and Scikit-Learn.",
        classifiers=['Programming Language :: Python :: 3'],
    )
