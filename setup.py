# StreetSpace
# See full license in LICENSE.txt

from setuptools import setup

# provide a long description using reStructuredText
long_description = """
**StreetSpace** is a package under development for measuring and analysing
streetscapes and street networks.
"""

# list of classifiers from the PyPI classifiers trove
classifiers = ['Development Status :: 2 - Pre-Alpha',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering :: GIS',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3.6']

with open('requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

# now call setup
setup(name='streetspace',
      version='0.1.0',
      description='Measure and analyse streetscapes and street networks',
      long_description=long_description,
      classifiers=classifiers,
      url='https://github.com/chesterharvey/StreetSpace',
      author='Chester Harvey',
      author_email='chesterharvey@gmail.com',
      license='MIT',
      platforms='any',
      packages=['streetspace'],
      install_requires=install_requires)