#!/usr/bin/env python3

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open('README.md') as f:
    long_description = f.read()


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering',
]

setup(
    name='ocean-C-lab',
    description='Tools for ocean carbon cycle calcs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    maintainer='[C]worthy',
    maintainer_email='mclong@ucar.edu',
    classifiers=CLASSIFIERS,
    url='https://ocean-c-lab.readthedocs.io',
    project_urls={
        'Documentation': 'https://ocean-c-lab.readthedocs.io',
        'Source': 'https://github.com/c-worthy-ocean/ocean-c-lab',
        'Tracker': 'https://github.com/c-worthy-ocean/ocean-c-lab/issues',
    },
    packages=find_packages(exclude=('tests',)),
    package_dir={'ocean-c-lab': 'ocean-c-lab'},
    include_package_data=True,
    install_requires=install_requires,
    license='Apache 2.0',
    zip_safe=False,
    entry_points={},
    keywords='c-worthy-ocean',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
)
