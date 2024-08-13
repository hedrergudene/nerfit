from setuptools import setup, find_packages

setup(
    name="nerfit",
    version="0.1.0",
    description="A Named Entity Recognition package.",
    author="Antonio Zarauz Moreno",
    author_email="hedrergudene@gmail.com",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "transformers==4.44.0",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "sentence_transformers==3.0.1",
        "litellm==1.43.7",
        "safetensors==0.4.4",
        "fire==0.6.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
