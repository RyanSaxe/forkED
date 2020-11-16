#formatting for setup file taken from objax repository
# as seen here: https://github.com/google/objax/blob/master/setup.py
import re

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

README_FILE = 'README.md'
REQUIREMENTS_FILE = 'requirements.txt'
VERSION_FILE = 'forkED/_version.py'
VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')

version = r.group(1)
long_description = open(README_FILE, encoding='utf-8').read()
install_requires = [str(r) for r in parse_requirements(open(REQUIREMENTS_FILE, 'rt'))]

setup(
    name='forkED',
    version=version,
    description='ForkED is a Novel Encoder Decoder architecture for end-to-end learning of embeddings with hierarchical components',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Ryan Saxe',
    author_email='ryancsaxe@gmail.com',
    url='https://github.com/RyanSaxe/forkED',
    packages=find_packages(),
    install_requires=install_requires,
)