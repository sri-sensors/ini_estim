from setuptools import setup, find_packages
import os
import glob

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    __license__ = f.read()

REQUIRED_PACKAGES = [
    'scipy>=1.3.1',
    'numpy',
    'matplotlib>=3.1.1',
    'attrs',
    'pandas',
    'scikit-learn',
    'jax',
    'tqdm',
    'wheel',
    'torch',
    'PyWavelets'
]

setup(
    name='ini_estim',
    version='0.1',
    description='Code for optimizing electrical stimulation of nerve fibers with mutual information',
    long_description=readme,
    authors=['Lauren Grosberg', 'Erik Matlin', 'David Stoker'],
    author_email='david.stoker@sri.com',
    license=__license__,
    install_requires=REQUIRED_PACKAGES,
    tests_require=['pytest'],
    packages=find_packages(exclude=('test', 'docs')),
    entry_points = {
        'console_scripts': [
            'train_mnist_encoder=ini_estim.scripts.deprecated.train_mnist_encoder:main',
            'train_encoder=ini_estim.scripts.deprecated.train_encoder:main',
            'train_decoder=ini_estim.scripts.deprecated.train_decoder:main',
            'train_autoencoder=ini_estim.scripts.deprecated.train_autoencoder:main',
            'plot_checkpoint=ini_estim.scripts.deprecated.plot_training_checkpoint:main',
            'ini_estim_train=ini_estim.scripts.train:main',
            'ini_estim_config=ini_estim.scripts.config:main',
            'ini_estim=ini_estim.scripts.main:main'
            ],
    }

)