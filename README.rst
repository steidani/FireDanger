###########################
FireDanger - Forest Fire Danger 
###########################
==================================================================================
Calculation of indices for forest fire risk assessment in weather and climate data
==================================================================================

!!IMPORTANT!! This tool is still under development and not yet intented for use

FireDanger is a Python package intended to simpify the process of analysing forest fire (and drought) in time series and gridded weather and climate datasets. It is built on top of `xarray`_.  
It contains implementations of several popular fire danger and drought indices:

- Canadian Fire Weather Index System (FWI) including all its 6 components

Coming up next:
- Angström Index
- Baumgartner Index
- Nesterov index
- Munger Drought index
- Fuel Moisture Index
- Fosberg Fire Weather Index
- Keetch-Byram Drought Index
- McArthur Mark 5 Forest Fire Danger Index

The package is developed as part of the project "Waldbrandmanagement auf der Alpennordseite" by the Canton of Bern, the `University of Bern <https://www.geography.unibe.ch/about_us/staff/dr_steinfeld_daniel/index_eng.html>`_ and the `Wyss Academy for Nature <https://www.wyssacademy.org/>`_.  
A big "thank you" to the `Swiss federal institute of forest, snow and landscape research WSL <https://www.wsl.ch/en/index.html>`_ for providing a public `WIKI <https://wikifire.wsl.ch/tiki-index.html>`_ with reference information on the mostly used fire weather indices.

..
  References
.. _xarray: https://xarray.pydata.org/en/stable/


Be aware that this is a free scientific tool in continous development, then it may not be free of bugs. Please report any issue on the GitHub portal.

============
Installation
============

Make sure you have the required dependencies (for details see docs/environment.yml):

- xarray
- pandas
- numpy
- netCDF4
- numba
- (for plotting on geographical maps: matplotlib and cartopy)
- (for parallel computing: dask)
 
To install the development version (master), do:

.. code:: bash

    pip install git+https://github.com/steidani/firedanger.git

Copy from Github repository
---------------------------

Copy/clone locally the latest version from FireDanger:

.. code-block:: bash

    git clone git@github.com:steidani/firedanger.git /path/to/local/firedanger
    cd path/to/local/firedanger

Prepare the conda environment:

.. code-block:: bash

    conda create -y -q -n firedanger_dev python=3.9.4 pytest
    conda env update -q -f environment.yml -n firedanger_dev

Install firedanger in development mode in firedanger_dev:

.. code-block:: bash

    conda activate firedanger_dev
    pip install -e .

Run the tests:

.. code-block:: bash

    python -m pytest



