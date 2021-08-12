from setuptools import setup
from setuptools import find_packages

__version__ = '0.1.0'

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='bodypartregressor',
      version='0.1.0',
      description='Body Part Regression',
      author='Ke Yan (primary), Daniel C Elton',
      author_email='daniel.elton@nih.gov',
      include_package_data=True, #include  MANIFEST.in
      classifiers=[
        'Programming Language :: Python :: 3.7.7',
	  ],
      install_requires=['easydict', 'nibabel', 'tqdm'],
      packages=find_packages(),
      zip_safe=False)
