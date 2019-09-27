#!/usr/bin/env python


import os
import sys

from setuptools import setup


setup(
    name='mathematics',
    version='0.1.0',
    description='Mathematics',
    keywords='mathematics',
    url='https://github.com/cowboysmall/mathematics',


    author='Jerry Kiely',
    author_email='jerry@cowboysmall.com',
    license='MIT',


    packages=['maths'],
    install_requires=[],


    zip_safe=False,


    classifiers=[
        'Development Status :: 6 - Mature',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
    ],


    test_suite='nose.collector',
    tests_require=['nose'],
)
