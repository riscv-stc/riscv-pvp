from setuptools import setup, find_packages
import glob

setup(
    name='rvpvp',
    version='0.1.0',
    packages=find_packages(),
    package_data = {
        'rvpvp': ['*', '*/*', '*/*/*', '*/*/*/*/*'],
    },
    install_requires=[
        'Click',
        'rich',
        'jax',
        'jaxlib',
        'PyYAML'
    ],
    entry_points={
        'console_scripts': [
            'rvpvp = rvpvp.main:cli',
        ],
    },
)
