#!/usr/bin/env python
# S. Kaan Cetindag - February 2021
from setuptools import setup
from setuptools.command.install import install


setup(
    name='bonin-wheel',
    version='1.0',
    author='S. Kaan Cetindag',
    author_email='cetindag.kaan@gmail.com',
    packages=['behavior_python'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description='Behavior Analysis Tool\n Disclaimer: For now works for wheel behaviour, more to come...',
    entry_points = {
        'console_scripts': [
          'detectionSession = behavior_python.detection.wheelDetectionSession:main'
        ]
        },
)