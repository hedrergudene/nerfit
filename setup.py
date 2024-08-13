from setuptools import setup, find_packages

setup(
    name="nerfit",
    version="0.1.0",
    description="nerFit: Few-shot entity recognition representation learning",
    author="Antonio Zarauz Moreno",
    author_email="hedrergudene@gmail.com",
    packages=find_packages(where='src'),  # Finds the packages under 'src'
    package_dir={'': 'src'},  # Maps the root package to 'src'
    install_requires=[
        "transformers==4.44.0",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "sentence_transformers==3.0.1",
        "litellm==1.43.7",
        "safetensors==0.4.4",
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
