#!/usr/bin/env python
# S. Kaan Cetindag - February 2021
from setuptools import setup
from setuptools.command.install import install


setup(
    name='boninbehavior',
    version='1.0',
    author='S. Kaan Cetindag',
    author_email='cetindag.kaan@gmail.com',
    packages=['behavior_python'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description='Behavior Analysis Tool\n',
    entry_points = {
        'console_scripts': [
          'parsemouse = behavior_python.core.mouse:main',
          'session = behavior_python.core.session_launcher:main',
          'dashboard = behavior_python.plotters.bokeh_plot.launcher:main'
        ]
        },
)