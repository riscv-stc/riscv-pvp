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
        'Click==8.0.4',
        'rich==10.0.0',
        'jax==0.2.17',
        'jaxlib==0.1.69',
        'PyYAML==6.0'
    ],
    entry_points={
        'console_scripts': [
            'rvpvp = rvpvp.main:cli',
        ],
    },
)
