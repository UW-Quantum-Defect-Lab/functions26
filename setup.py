import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="functions26",
    version="0.1.5.0",
    author="Vasilis Niaouris",
    author_email="vasilisniaouris@gmail.com",
    description="Supporting functions for Fu lab 26 room",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vasilisniaouris/functions26",
    packages=setuptools.find_packages(),
    package_dir={'functions26': 'functions26'},
    package_data={'functions26': ['external_data/*.csv']},
    install_requires=['collections-extended',
                      'lmfit',
                      'matplotlib',
                      'nidaqmx',
                      'numpy',
                      'pandas',
                      'py3-validate_email',
                      'py3dns',
                      'pydaqmx',
                      'pyserial',
                      'pyvisa',
                      'pyqt5',
                      'pyqtgraph',
                      'scipy',
                      'seaborn',
                      'sif_reader @ https://github.com/fujiisoup/sif_reader/tarball/master#egg=package-1.0',
                      'spinmob',
                      'windfreak'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
