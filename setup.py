from setuptools import setup, find_packages

setup(
    name="counterfactual_methods",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'networkx>=2.5',
        'matplotlib>=3.3.0',
        'xgboost>=1.3.0',  # If using XGBoost
        'tensorflow>=2.4.0'  # If using neural networks
    ],
    python_requires='>=3.8',
    author="Waruni Kekulandara",
    author_email="warunilkekulandara20@gmail.com",
    description="Counterfactual explanation methods implementation",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/warunik/Counterfactual_Methods",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)