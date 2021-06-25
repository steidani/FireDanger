"""

Description
-----------

 The firedanger package provides classes and function to read, write, plot and
 analyze forest fire danger in weather and climate data (gridded and timeseries).


Content
-------

 The following classes are available:

 firedanger:      To create a firedanger object with functions to calculate firedanger indices.


Examples
--------

>>> filename = 'data.nc' or 'data.csv'
>>> fire = firedanger()
>>> fire.read_nc(filename.nc) or fire.read_nc(filename.csv)

"""
from .firedanger import firedanger  # noqa