import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
    name="TinyImageNetChallenge",
    version="1.0.0",
    description="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)