This package has been created for use in the Fu lab at UW. It contains simple functions that make our every day coding
easier.

_________________________________

To simply install:
    (open cmd with administrator privileges)
    (cd to the directory setup.py resides (might need to change the disk from C:\ to e.g. D:\ by typing d: ))
    pip install dist/functions26-0.1.5.0.tar.gz

An additional dependency might need to be installed with the commands:
    pip install ./downloaded_dependencies/sif_reader-master.zip
    pip install pyserial

_________________________________

When you make changes here, make sure to run the following commands to update the package:
    (open cmd with administrator privileges)
    (cd to the directory setup.py resides (might need to change the disk from C:\ to e.g. D:\ by typing d: ))
    python -m setup.py sdist bdist_wheel
    python -m pip install dist/functions26-0.1.5.0.tar.gz
    python -m setup.py sdist bdist_wheel & python -m pip install dist/functions26-0.1.5.0.tar.gz 
_________________________________

The files to modify are in the functions26
The old versions are in the useful_functions_package_outdated folder (parallel to functions26_package)
    the recent old versions are under the folder dist with name with name useful_functions-[version].tar.gz for versions 0.0.3 and above
    the even older versions are under folder dist, with name useful_functions-[version].tar.gz for versions 0.0.3 and below

_________________________________
For Windows:
	Sometimes you should use py, instead of python in a command
	Sometimes you need to add -m (i.e. py -m pip install ...)