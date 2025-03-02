from setuptools import setup, find_packages

setup(
    name='diffusion_mnist',
    version='0.1.0',
    author='yukoga',
    license='Apache-2.0',
    packages=find_packages(where='diffusion_mnist'),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'matplotlib'
    ],
)
