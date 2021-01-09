import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="functions26",
    version="0.0.6",
    author="Vasilis Niaouris",
    author_email="vasilisniaouris@gmail.com",
    description="Supporting functions for Fu lab 26 room",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vasilisniaouris/functions26",
    packages=setuptools.find_packages(),
    install_requires=['collections-extended',
                      'matplotlib',
                      'nidaqmx',
                      'numpy',
                      'pandas',
                      'py3-validate_email',
                      'py3dns',
                      'pydaqmx',
                      'pyserial',
                      'pyvisa',
                      'sif-reader',
                      'scipy',
                      'seaborn',
                      'windfreak'],
    dependency_links=['https://github.com/fujiisoup/sif_reader//tarball/master#egg=package'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
