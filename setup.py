from setuptools import setup, find_packages

setup(
    name='stock_sense',  
    version='0.1',  
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'yfinance',
        'streamlit',  
    ],
    entry_points={
        'console_scripts': [
            'stock-sense=app:main',  
        ],
    },
)
