from setuptools import setup, find_packages

setup(
    name="rtl_timing_analysis",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "networkx",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "joblib",
        "numpy"
    ],
    author="Your Name",
    description="AI-powered RTL timing analysis tool",
    license="MIT"
)
