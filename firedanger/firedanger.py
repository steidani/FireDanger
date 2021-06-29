#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Description
-----------

 The firedanger package provides classes and function to read, write, plot and
 analyze forest fire danger in weather and climate data (gridded and timeseries).


Content
-------

 The following classes are available:

 firedanger:      To create a firedanger object with functions to calculate forest fire indices.


Examples
--------

>>> filename = 'data.nc' or 'data.csv'
>>> fire = firedanger() 
>>> fire.read_nc(filename.nc) or fire.read_csv(filename.csv) 

Author
--------

@author: steidani (Daniel Steinfeld; daniel.steinfeld@alumni.ethz.ch)


TO DO
    - __init__: csv file should not raise error.
    - COSMO: lat and lon are in swiss coordinate named x_1 and y_1: us ds.coords to find lat lon?
    - cannot yet handle dask: in future use .data instead of .values?
    - xarray apply_ufunc: https://xarray.pydata.org/en/stable/examples/apply_ufunc_vectorize_1d.html
      https://stackoverflow.com/questions/57552588/apply-function-along-time-dimension-of-xarray

"""

# =======
# import packages

# data
import numpy as np
import xarray as xr
from scipy import ndimage
import pandas as pd
from numpy.core import datetime64

import firedanger.indices as indices

# logs
import logging
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# parallel computing
try:
    import dask
except:
    logger.warning("Dask is not installed in your python environment. Xarray Dataset parallel computing will not work.")


# plotting
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
except:
    logger.warning("Matplotlib and/or Cartopy is not installed in your python environment. Xarray Dataset plotting functions will not work.")


# =============================================================================
# firedanger class
# =============================================================================

class firedanger(object):
    """
    firedanger class
    Author : Daniel Steinfeld, University of Bern , 2021
    """

    # number of instances initiated
    num_of_firedanger = 0

    def __init__(self, filename="", ds=None, **kwargs):
        """The constructor for firedanger class. Initialize a firedanger instance.
        
        If filename is given, try to load it directly.
        Arguments to the load function can be passed as key=value argument.

        Parameters
        ----------
            filename : string
                Datapath + filename
            ds : dataset
                xarray dataset

        """
        if not filename:
            if ds is None:
                self.ds = None
            else:
                self.ds = ds
            return
        
        try:
            self.ds = None
            self.read_nc(filename, **kwargs)
        except Exception:
            try:
                self.ds = None
                self.read_csv(filename, **kwargs)
            except IOError:
                raise IOError("Unkown fileformat. Known formats are netcdf or csv.")

        firedanger.num_of_firedanger += 1
    
    def __repr__(self):
        try:
            string = "\
            Xarray dataset with {} time steps. \n\
            Available fields: {}".format(
                self.ntime, ", ".join(self.variables)
            )
        except AttributeError:
            # Assume it's an empty fire()
            string = "\
            Empty firedanger container.\n\
            Hint: use read_nc() or read_csv() to load data."
        return string

    #def __str__(self):
    #    return 'Class {}: \n{}'.format(self.__class__.__name__, self.ds)
  
    def __len__(self):
        return len(self.ds)
    
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.ds, attr)

    
    def __getitem__(self, key):
        return self.ds[key]

    @property
    def ntime(self):
        """Return the number of time steps"""
        if len(self.ds.dims) == 1:
            # only one dimension "time"
            return self.ds.dims[self._get_name_time()]
        elif len(self.ds.dims) != 3:
            logger.warning(
                "\nBe careful with the dimensions, "
                "you want dims = 3 and shape:\n"
                "(time, latitude, longitude)"
            )
            return self.ds.dims[self._get_name_time()]
        return self.ds.dims[self._get_name_time()]

    @property
    def variables(self):
        """Return the names of the variables"""
        return list(self.ds.data_vars)
    
    @property
    def dimensions(self):
        """Return the names of the dimensions"""
        return list(self.ds.dims)
    
    @property
    def grid(self):
        """Return the number of longitude and latitude grid"""
        if len(self.ds.dims) == 1:
            logger.warning(
                "\nNo grid "
                "for time series data."
            )
            return None
        if len(self.ds.dims) != 3:
            logger.warning(
                "\nBe careful with the dimensions, "
                "you want dims = 3 and shape:\n"
                "(time, latitude, longitude)"
            )
            return None
        string = "\
        latitude: {} \n\
        longitude: {}".format(
            self.ds.dims[self._get_name_latitude()], self.ds.dims[self._get_name_longitude()]
        ) 
        print(string)

    @property
    def dataset(self):
        """Return the dataset"""
        return self.ds

# ----------------------------------------------------------------------------
# Read / Import / Save data
    
    def read_nc(self, filename, **kwargs):
        """
        Reads a netcdf file into a xarray dataset.
        
        Parameters
        ----------
            filename : string
                Valid path + filename
        """
        if self.ds is None:
            self.ds = xr.open_dataset(filename, **kwargs)
            logger.debug('read: {}'.format(self.__str__))
        else:
            errmsg = 'contrack() is already set!'
            raise ValueError(errmsg)
            
    def read_xarray(self, ds):
        """
        Read an existing xarray data set.
        
        Parameter:
        ----------
            ds: data set
                Valid xarray data set.
        """
        if self.ds is None:
            if not isinstance(ds, xr.core.dataset.Dataset):
                errmsg = 'ds has to be a xarray data set!'
                raise ValueError(errmsg)
            self.ds = ds
            logger.debug('read_xarray: {}'.format(self.__str__))
        else:
            errmsg = 'firedanger() is already set!'
            raise ValueError(errmsg)

    def read_csv(self, filename, time_name="time", **kwargs):
        """
        Reads a csv file into a xarray dataset.
        
        Parameters
        ----------
            filename : string
                Valid path + filename
            time_name : string in format YYYYMMDD (e.g. 20200130)
                Name of time dimension/column. The default is "time".
        """
        if self.ds is None:
            self.ds = pd.read_csv(filename, **kwargs).set_index([time_name]).to_xarray()
            self.ds[time_name] = pd.to_datetime(self.ds[time_name].values, format='%Y%m%d', errors='coerce')
            logger.debug('read: {}'.format(self.__str__))
        else:
            errmsg = 'firedanger() is already set!'
            raise ValueError(errmsg)
 
# ----------------------------------------------------------------------------
# Set up / Check dimensions
   
    def set_up(self,
               time_name=None,
               longitude_name=None,
               latitude_name=None,
               force=False,
               write=True
    ):
        """
        Prepares the dataset for contour tracking. Does consistency checks
        and tests if all required information is available. Sets (automatically 
        or manually) internal variables and dimensions.

        Parameters
        ----------
            time_name : string, optional
                Name of time dimension. The default is None.
            longitude_name : string, optional
                Name of x or longitude dimension. The default is None.
            latitude_name : string, optional
                Name of y or latitude dimension. The default is None.
            force=False: bool, optional 
                Skip some consistency checks.
            write=True: bool, optional
                Print name of dimensions.

        Returns
        -------
            None.

        """

        # set dimensions
        if time_name is None:
            self._time_name = self._get_name_time()  
        else:
            self._time_name = time_name
        if longitude_name is None:
            self._longitude_name = self._get_name_longitude()
        else:
            self._longitude_name = longitude_name
        if latitude_name is None:
            self._latitude_name = self._get_name_latitude()
        else:
            self._latitude_name = latitude_name

        # set resolution
        if (self._longitude_name and self._latitude_name) is not None:
            self._dlon =  self._get_resolution(self._longitude_name, force=force)
            self._dlat =  self._get_resolution(self._latitude_name, force=force)
        else: # time series data
            self._dlon =  None
            self._dlat =  None

        if self._time_name is not None:
            if self.ntime > 1:
                self._dtime = self._get_resolution(self._time_name, force=force)
            else: # only one time step
                self._dtime = None

        # print names    
        if write:
            logger.info(
                "\n time: '{}'\n"
                " longitude: '{}'\n"
                " latitude: '{}'\n".format(
                self._time_name, 
                self._longitude_name,
                self._latitude_name)
            )

    
    def _get_name_time(self):
        """
        check for 'time' dimension and return name
        """
        # check unit
        for dim in self.ds.dims:
            if (('units' in self.ds[dim].attrs and
                'since' in self.ds[dim].attrs['units']) or 
                ('units' in self.ds[dim].encoding and
                 'since' in self.ds[dim].encoding['units']) or
                dim in ['time']):
                return dim
        # check dtype
        for dim in self.ds.variables:
            try:
                var = self.ds[dim].data[0]
            except IndexError:
                var = self.ds[dim].data
            if isinstance(var, datetime64):
                return dim   
        # no 'time' dimension found
        logger.warning(
            "\n 'time' dimension (dtype='datetime64[ns]') not found."
        )
        return None     


    def _get_name_longitude(self):
        """
        check for 'longitude' dimension and return name
        """
        for dim in self.ds.dims:
            if (dim in ['lon', 'lon_1', 'longitude', 'x', 'x_1'] or 
               ('units' in self.ds[dim].attrs and
               self.ds[dim].attrs['units'] in ['degree_east', 'degrees_east'])):
               return dim
        # no 'longitude' dimension found
        logger.warning(
            "\n 'longitude' or 'x' dimension not found."
        )
        return None


    def _get_name_latitude(self):
        """
        check for 'latitude' dimension and return name
        """
        for dim in self.ds.dims:
            if (dim in ['lat', 'lat_1', 'latitude', 'y', 'y_1'] or 
               ('units' in self.ds[dim].attrs  and
                self.ds[dim].attrs['units'] in ['degree_north', 'degrees_north'])):
                return dim
        # no 'latitude' dimension found
        logger.warning(
            "\n 'latitude' or 'y' dimension not found."
        )
        return None
            
    def _get_resolution(self, dim, force=False):
        """
        set spatial (lat/lon) and temporal (time) resolution
        """
        # time dimension in hours
        if dim == self._time_name:
            try:
                var = self.ds[dim].to_index()
                delta = np.unique((
                    self.ds[dim].to_index()[1:] - 
                    self.ds[dim].to_index()[:-1])
                    .astype('timedelta64[h]')
                )
            except AttributeError:  # dates outside of normal range
                # we can still move on if the unit is "days since ..."
                if ('units' in self.ds[dim].attrs and
                    'days' in self.ds[dim].attrs['units']):
                    var = self.ds[dim].data
                    delta = np.unique(var[1:] - var[:-1])
                else:
                    errmsg = 'Can not decode time with unit {}'.format(
                        self.ds[dim].attrs['units'])
                    raise ValueError(errmsg)
        # lat/lon dimension in Degree
        else:
            delta = abs(np.unique((
                self.ds[dim].values[1:] - 
                self.ds[dim].values[:-1])
            ))
        # check resolution
        if len(delta) > 1:
            errmsg = 'No regular grid found for dimension {}.\n\
            Hint: use set_up(force=True).'.format(dim)
            if force and dim != self._time_name:
                logging.warning(errmsg)
                logmsg = ' '.join(['force=True: using mean of non-equidistant',
                                   'grid {}'.format(delta)])
                logging.warning(logmsg)
                delta = round(delta.mean(), 2)
            else:
                if dim == self._time_name:
                    logging.warning(errmsg)
                else:
                    raise ValueError(errmsg)
        elif delta[0] == 0:
            errmsg = 'Two equivalent values found for dimension {}.'.format(
                dim)
            raise ValueError(errmsg)
        elif delta[0] < 0:
            errmsg = ' '.join(['{} not increasing. This should',
                                   'not happen?!']).format(dim)
            raise ValueError(errmsg)
            
        return delta
        
        
# ----------------------------------------------------------------------------
# calculations
  
    def calc_windspeed(self,
                       u,
                       v,

    ):
        """ Compute the wind speed from u and v-components.

        Note: Does not yet support dask array. xarray.Dataset.values are taken for calculations.
            - using .chunks({"time":-1}) creates a view (?) and you cannot assign values 
        
        Parameters:
        ----------
            u: string
               Name of wind component in the X (East-West) direction
            v: string
               Name of wind component in the Y (North-South) direction
            
        Returns
        -------
            ds: xarray dataset
                Speed of the wind [m/s]
        """
        # create new variable wind     
        self.ds['wind'] = xr.Variable(
            self.ds[u].dims,
            xr.ufuncs.sqrt(self.ds[u]**2 + self.ds[v]**2).data,  
            attrs={
                'units': self.ds[u].attrs['units'],
                'long_name': 'windspeed',
                'standard_name': 'wind',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}.'])
                    .format(u, v)}
        )
        logger.info('Calculating wind speed... DONE')

  
    def calc_canadian_fwi(self, 
                  temp, 
                  precip, 
                  hum, 
                  wind
    ):
        """ Calculate the Canadian Fire Weather Index 

        Parameters
        ----------
        t: array
            Noon temperature [C].
        p : array
            rainfall amount in open over previous 24 hours, at noon [mm].
        w : array
            Noon wind speed [m/s].
        h : array
            Noon relative humidity [%].
        
        Returns
        -------
        ds: xarray dataset
            Components of the Canadian Fire Weather Index: ffmc, dmc, dc, isi, bui, fwi [dimensionless]

        To DO:
        - latitude array as input --> does not yet work in indices._day_length
          currently, lat = 47 (Switzerland)
        - initial values of ffmc, dmc and dc
        - cannot handle dask array yet: current solution is to load dataset
        """

        # Set up dimensions
        logger.info("Set up dimensions...")
        if hasattr(self, '_time_name'):
            # print names       
            logger.info(
                "\n time: '{}'\n"
                " longitude: '{}'\n"
                " latitude: '{}'\n".format(
                self._time_name, 
                self._longitude_name,
                self._latitude_name)
            )
            pass
        else:
            self.set_up()

        # initiate output and ffmc0
        ffmc = 85.0
        dmc = 6.0 
        dc = 15.0
        # THIS IS QUICK AND UGLY!!! 
        out_ffmc = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        out_dmc = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        out_dc = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        out_isi = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        out_bui = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        out_fwi = xr.full_like(self.ds[temp], np.nan, dtype=np.float32)
        # if dask array, need to load into memory (this takes a little time); dask array in xarray does not yet support item assignment. See issue: https://github.com/pydata/xarray/issues/5171)
        out_ffmc.load()
        out_dmc.load()
        out_dc.load()
        out_isi.load()
        out_bui.load()
        out_fwi.load()

        # loop over time
        # calculate indices at each timestep
        for i_time in range(self.ds.dims[self._time_name]): 
            currentstep = self.ds[self._time_name].isel(**{self._time_name: i_time}).dt.strftime('%Y%m%d_%H').values
            
            ffmc = indices.ffmc(self.ds[temp].isel(**{self._time_name: i_time}).values, 
                                self.ds[precip].isel(**{self._time_name: i_time}).values, 
                                self.ds[wind].isel(**{self._time_name: i_time}).values, 
                                self.ds[hum].isel(**{self._time_name: i_time}).values, 
                                ffmc)
            
            dmc = indices.dmc(self.ds[temp].isel(**{self._time_name: i_time}).values, 
                                self.ds[precip].isel(**{self._time_name: i_time}).values, 
                                self.ds[hum].isel(**{self._time_name: i_time}).values, 
                                self.ds[self._time_name].isel(**{self._time_name: i_time}).dt.month.values,
                                47, # self.ds[self._latitude_name].values
                                dmc)

            dc = indices.dc(self.ds[temp].isel(**{self._time_name: i_time}).values, 
                                self.ds[precip].isel(**{self._time_name: i_time}).values, 
                                self.ds[self._time_name].isel(**{self._time_name: i_time}).dt.month.values, 
                                47, # self.ds[self._latitude_name].values
                                dc)

            isi = indices.isi(self.ds[wind].isel(**{self._time_name: i_time}).values, 
                              ffmc)
            
            bui = indices.bui(dmc, 
                              dc)
            
            fwi = indices.fwi(isi, 
                              bui)

            
            # item assignment
            out_ffmc[dict(**{self._time_name: i_time})] = ffmc
            out_dmc[dict(**{self._time_name: i_time})] = dmc
            out_dc[dict(**{self._time_name: i_time})] = dc
            out_isi[dict(**{self._time_name: i_time})] = isi
            out_bui[dict(**{self._time_name: i_time})] = bui
            out_fwi[dict(**{self._time_name: i_time})] = fwi
        out_ffmc.close()
        out_dmc.close()
        out_dc.close()
        out_isi.close()
        out_bui.close()
        out_fwi.close()

        # create new variable ffmc     
        self.ds['ffmc'] = xr.Variable(
            self.ds[temp].dims,
            out_ffmc.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Fine fuel moisture code',
                'standard_name': 'ffmc',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}, {}, {}.'])
                    .format(temp, precip, hum, wind)}
        )

        # create new variable dmc     
        self.ds['dmc'] = xr.Variable(
            self.ds[temp].dims,
            out_dmc.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Duff moisture code',
                'standard_name': 'dmc',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}, {}.'])
                    .format(temp, precip, hum)}
        )

        # create new variable dc     
        self.ds['dc'] = xr.Variable(
            self.ds[temp].dims,
            out_dc.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Drought code',
                'standard_name': 'dc',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}.'])
                    .format(temp, precip)}
        )

        # create new variable isi     
        self.ds['isi'] = xr.Variable(
            self.ds[temp].dims,
            out_isi.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Initialize spread index',
                'standard_name': 'isi',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}.'])
                    .format(wind, "ffmc")}
        )

        # create new variable bui     
        self.ds['bui'] = xr.Variable(
            self.ds[temp].dims,
            out_bui.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Build-up index',
                'standard_name': 'bui',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}.'])
                    .format("dmc", "dc")}
        )

        # create new variable fwi     
        self.ds['fwi'] = xr.Variable(
            self.ds[temp].dims,
            out_fwi.values,  
            attrs={
                'units': 'dimensionless',
                'long_name': 'Fire weather index',
                'standard_name': 'fwi',
                'history': ' '.join([
                    'Calculated from input variables: {}, {}.'])
                    .format("isi", "bui")}
        )
        # chunk it
        #self.ds['ffmc'] = self.ds['ffmc'].chunk(chunks = {self.ds[temp].chunks})
        logger.info('Calculating Canadian Fire Weather Index (ffmc, dmc, dc, isi, bui, fwi)... DONE')