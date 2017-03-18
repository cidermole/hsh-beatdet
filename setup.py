from distutils.core import setup

setup(
    name='hsh-beatdet',
    version='0.0.1',
    packages=['hsh_beatdet']
    #ext_modules=extensions
)

# run as:
# python setup.py build_ext --inplace [--rpath=...]

# python setup.py develop

# pip2 install --editable .
