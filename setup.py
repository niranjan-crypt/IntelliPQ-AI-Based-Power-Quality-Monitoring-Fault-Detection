from setuptools import setup, find_packages

setup(
    name="intellipq",
    version="1.0.0",
    description="IntelliPQ: AI-Based Power Quality Monitoring & Fault Detection System",
    author="Niranjan Krishnakumar",
    author_email="niranjankrishnakumar2005@gmail.com",
    url="https://github.com/niranjan-crypt/IntelliPQ-AI-Based-Power-Quality-Monitoring-Fault-Detection",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
        "pandas>=1.5.0",
        "streamlit>=1.25.0",
    ],
    extras_require={
        "ai": ["tensorflow>=2.10.0"],
        "dev": ["pytest>=7.0.0", "black", "flake8"],
    },
    classifiers=[
        "Topic :: Scientific/Engineering :: Interface Engine/Protocol Translator",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
