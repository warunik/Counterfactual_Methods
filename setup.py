from setuptools import setup, find_packages

setup(
    name="counterfactual_methods",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add other dependencies here
    ],
)