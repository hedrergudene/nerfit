from setuptools import setup, find_packages

setup(
    name="nerfit",
    version="0.1.0",
    description="nerFit: Few-shot entity representation learning",
    author="Antonio Zarauz Moreno",
    author_email="hedrergudene@gmail.com",
    url="https://github.com/hedrergudene/nerfit",
    packages=["nerfit"],
    install_requires=[
        "transformers==4.44.0",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "evaluate==0.4.2",
        "sentence_transformers==3.0.1",
        "litellm==1.43.7",
        "safetensors==0.4.4",
        "numpy==1.26.0",
        "fire==0.6.0"
    ],
    extras_require={
        'dev': [
            'pytest>=8.3.2',
            'flake8>=6.0.0'
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
