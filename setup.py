<<<<<<< HEAD
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("ml_ext_pyscipopt.pyx"),
)
=======
from setuptools import setup, find_packages


setup(
    name="learn2branch",
    version="1.0.0",
    author="Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin, Andrea Lodi",
    install_requires=["numpy", "scipy"],
    packages=["learn2branch"],
    package_dir={"learn2branch": "."},
)
>>>>>>> upstream/master
