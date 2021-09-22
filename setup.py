import re
from setuptools import setup

VERSIONFILE="wfsim/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    __version__ = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open("README.rst", 'r') as fh:
    long_description = fh.read()

setup(
    name='wfsim',
    version=__version__,
    author='Josh Meyers',
    author_email='jmeyers314@gmail.com',
    url='https://github.com/jmeyers314/wfsim',
    description="Wavefront Donut Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['wfsim'],
    package_dir={'wfsim': 'wfsim'},
    install_requires=['numpy', 'batoid'],
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
