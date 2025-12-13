from setuptools import setup, find_packages

setup(
    name="mlops-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'databricks-cli>=0.18.0',
        'click>=8.1.7',
        'python-dotenv>=1.0.0',
    ],
)
