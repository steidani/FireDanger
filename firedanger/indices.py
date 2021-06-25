#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""    
==================== 
Fire Weather Indices
==================== 

@author: steidani (Daniel Steinfeld; daniel.steinfeld@giub.unibe.ch)

This module defines the
  - Angström index
  - Baumgartner Index (ToDo)
  - Nesterov index
  - Munger Drought index
  - Fuel moisture index an Modification with drought factor (ToDo)
  - Fosberg fire weather index (ToDo)
  - Keetch-Byram drought index 
  - McArthur Mark 5 forest fire danger index (ToDo)
  - ffmc: Fine Fuel Moisture Code (canadian index)
  - dmc: Duff moisture code (canadian index)
  - dc: Drought Code (canadian index)
  - isi: Initial Spread Index (canadian index)
  - bui: Buildup Index (canadian index)
  - fwi: Fire Weather Index (canadian index)
  - dsr: Daily Severity Rating (canadian index)
indices used for forest fire danger assessment.

TO DO
-----
ACHTUNG MIT NUMPY POWER! Does not accept negative exponents!

References
----------
Overview different Indices:
  - WSL Fire Weather Indices WIKI: https://wikifire.wsl.ch/tiki-index515f.html?page=Introduction&structure=Fire
Angström Index:
  - Chandler et al., 1983. Fire in forestry. Bd. 1. New York: John Wiley and Sons. 
Baumgartner Index:
  - Baumgartner et al., 1967. Waldbrände in Bayern 1950 bis 1959. Mitteilungen aus der Staatsforstverwaltung Bayerns 36: 57-79. 
Munger Drought Index:
  - Munger, Graphic method of representing and comparing drought intensities. Monthly Weather Review 44: 642-643. 
Nestroc Index:
  - Nesterov, 1949. Combustibility of the forest and methods for its determination (in Russian). USSR State Industry Press. 
  - Chandler et al., 1983. Fire in forestry. Bd. 1. New York: John Wiley and Sons. 
Canadian Forest Fire Weather Index System (FWI):
  - van Wagner, 1987. Development and structure of the Canadian Forest Fire Weather Index (Forestry Tech. Rep. No. 35). Canadian Forestry Service, Ottawa, Canada.
  - van Wagner et al., 1985. Equations and FORTRAN program for the Canadian forest fire weather index system (Forestry Tech. Rep. No. 33). Canadian Forestry Service, Ottawa, Canada.
  - Wang, et al., 2015. Updated source code for calculating fire danger indices in the Canadian Forest Fire Weather Index System. INFORMATION REPORT NOR-X-424.
  - Cantin et al., Canadian Forest Fire Danger Rating System, R package, CRAN, https://cran.r-project.org/package=cffdrs
"""

# import modules
import math
import numpy as np
import pandas as pd
import xarray as xr
#import metpy.calc as mpcalc    # not yet used, but maybe in future
#from metpy.units import units
import datetime as datetime
from numba import jit, vectorize
from typing import Optional, Sequence, Union

# ===============
# Meteorological Variables
# ===============

def tdewpoint_from_relhum(t, h):
  """Compute the Dewpoint temperature.

  Parameters
  ----------
  t: array
    temperature [C].
  h : array
    relative humidity [%].
  
  Returns
  -------
  array
    Dewpoint temperature at current timestep.

  Example
  -------
  tdewpoint_from_relhum(20,50) == 9.270085985370075
  """
  A = 17.67
  B = 243.5
  alpha = ((A * t) / (B + t)) + math.log(h/100.0)
  return (B * alpha) / (A - alpha)
  
  #return mpcalc.dewpoint_from_relative_humidity(t* units.degC, h* units.percent).magnitude

# ===============
# Net Radiation
# ===============

def extra_radiation(date, lat):
  """ Compute Extraterrestrial radiation [MJ·m-2·d-1].
      Based on Eq. 21 in Allen et al. 1998.
      "top of the atmosphere radiation"

  Parameters
  ----------
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.

  Returns
  -------
  array
    Extraterrestrial radiation in [MJ·m-2·d-1].
  
  """  
  # solar constant
  sol_con = 0.082 # [MJ·m-2·min-1],
  # get Julian date: number of the day in the year between 1 (1 January) and 365 or 366 (31 December).
  day_of_year = datetime.datetime.strptime(date,'%Y%m%d').timetuple().tm_yday
  days_in_year = datetime.date(int(datetime.datetime.strptime(date,'%Y%m%d').year),12,31).timetuple().tm_yday
  # solar declination δ
  delta = 0.409 * np.sin((2 * np.pi * day_of_year / days_in_year)-1.39)
  # convert degrees to radians
  phi = np.deg2rad(lat)
  # sunset hour angle [radians]
  # Domain of arccos is -1 <= x <= 1 radians
  ws = np.arccos(np.min(np.max((-1 * np.tan(phi) * np.tan(delta), -1.0), 1.0)))
  # inverse relative distance Earth-Sun
  dr = 1 + 0.033 * np.cos(day_of_year * 2 * np.pi / days_in_year)

  tmp1 = 24 * 60 / np.pi
  tmp2 = ws * np.sin(phi) * np.sin(delta) 
  tmp3 = np.cos(phi) * np.cos(delta) * np.sin(ws)
  return tmp1 * sol_con * dr * (tmp2 + tmp3)

def solar_radiation_from_sunshine(sunshine, date, lat):
  """ Compute daily solar radiation from Sunshine duration using Angstrom formula after 
      Eqs. 34 and 35 in Allen et al. 1998.

  Parameters
  ----------
  sunshine: array
    Daily sunshine duration [h].
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.

  Returns
  -------
  array
    Daily solar radiation in [MJ·m-2·d-1].
  
  """  
  # Calculate maximum potential daylight hour
  daylighthour = daylighthour_FAO(date, lat)
  # calculate extraterrestrial radiation
  et_rad = extra_radiation(date, lat)
  # To Do: solar radiation value is constrained by the clear sky radiation: np.min(sol_rad, cs_rad)

  # 0.5 and 0.25 are default values of regression constants (Angstrom values)
  return (0.25 + 0.5 * sunshine / daylighthour ) * et_rad

def solar_radiation_from_temp(tmin, tmax, date, lat, krs=16):
  """ Compute daily solar radiation from temperature after 
      Eq. 50 in Allen et al. 1998.

  Parameters
  ----------
  tmin: array
    minimum daily temperature [C].
  tmax: array
    maximum daily temperature [C].
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.
  krs : scalar
    adjustment coefficient [deg C-0.5] for coastal (19) or interior (16) locations

  Returns
  -------
  array
    Daily solar radiation in [MJ·m-2·d-1].
  
  """  
  # calculate extraterrestrial radiation
  et_rad = extra_radiation(date, lat)

  return et_rad * krs * np.sqrt(tmax - tmin)

def clearsky_radiation(date, lat, altitude):
  """ Compute clear-sky solar radiation after Allen et al. 1998.

  Parameters
  ----------
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.
  altitude: scalar
    altitude of location.

  Returns
  -------
  array
    Clear-sky solar radiation in [MJ·m-2·d-1].
  
  """  
  et_rad = extra_radiation(date, lat) # extraterrestrial radiation
  return (0.00002 * altitude + 0.75) * et_rad

def net_in_sol_rad(sol_rad, albedo=0.23):
  """ Compute net incoming shortwave solar radiation after Eq. 33 in Allen et al. 1998.

  Parameters
  ----------
  sol_rad: array
    Solar radiation in [MJ·m-2·d-1]. Either calculated from sunshine duration or temperature.
  albedo : scalar 
    Proportion of gross incoming solar radiation that is reflected by the surface.
    Albedo can be as high as 0.95 for freshly fallen snow and as low as 0.05 for wet bare soil. 
    A green vegetation over has an albedo of about 0.20-0.25 (Allen et al, 1998).

  Returns
  -------
  array
    Net incoming solar radiation in [MJ·m-2·d-1].
  
  """  
  return (1 - albedo) * sol_rad

def net_out_lw_rad(tmin, tmax, h, date, lat, altitude, krs=16):
  """ Compute net outgoing longwave solar radiation after Eq. 39 in Allen et al. 1998.
  This is the net longwave energy (net energy flux) leaving the earth's surface. 
  It is proportional to the absolute temperature of the surface raised to the fourth power 
  according to the Stefan-Boltzmann law. However, water vapour, clouds, carbon dioxide and dust are 
  absorbers and emitters of longwave radiation. This function corrects the Stefan-
  Boltzmann law for humidity (using actual vapor pressure) and cloudiness
  (using solar radiation and clear sky radiation). The concentrations of all
  other absorbers are assumed to be constant.
  The output can be converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

  Parameters
  ----------
  tmin: array
    minimum daily temperature [C].
  tmax: array
    maximum daily temperature [C].
  h: array 
    relative humidity [%].
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.
  altitude: scalar
    altitude of location.
  krs : scalar
    adjustment coefficient [deg C-0.5] for coastal (19) or interior (16) locations

  Returns
  -------
  array
    Net outgoing longwave radiation [MJ m-2 day-1]. 

  """
  rho = 4.903 * np.power(10,-9) # Stefan Boltzmann constant
  es = 0.5 * (sat_vapour_pressure(tmin) + sat_vapour_pressure(tmax)) # saturation vapour pressure
  ea = es*h/100 # actual vapour pressure
  sol_rad = solar_radiation_from_temp(tmin, tmax, date, lat, krs=16) # Solar radiation
  cs_rad = clearsky_radiation(date, lat, altitude) # Clear sky radiation

  tmp1 = rho * ((np.power(tmax + 273.16, 4) + np.power(tmin + 273.16, 4)) / 2)
  tmp2 = 0.34 - (0.14 * np.sqrt(ea))
  tmp3 = 1.35 * np.min(1, (sol_rad / cs_rad)) - 0.35 # sol_rad is constrained by cs_rad
  return tmp1 * tmp2 * tmp3

def net_radiation(tmin, tmax, h, date, lat, altitude, sol_rad, krs=16, albedo=0.23):
  """ Compute daily net radiation [MJ m-2 day-1] based on Eq. 40 in Allen et al (1998).
      At the crop surface, assuming a grass reference crop.
      reference crop

  Parameters
  ----------
  tmin: array
    minimum daily temperature [C].
  tmax: array
    maximum daily temperature [C].
  h: array 
    relative humidity [%].
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.
  altitude: scalar
    altitude of location.
  krs : scalar
    adjustment coefficient [deg C-0.5] for coastal (19) or interior (16) locations
  sol_rad: array
    Solar radiation in [MJ·m-2·d-1]. Either calculated from sunshine duration or temperature.
  albedo : scalar
    Proportion of gross incoming solar radiation that is reflected by the surface.
    Albedo can be as high as 0.95 for freshly fallen snow and as low as 0.05 for wet bare soil. 
    A green vegetation over has an albedo of about 0.20-0.25 (Allen et al, 1998).

  Returns
  -------
  array
    Net outgoing longwave radiation [MJ m-2 day-1]. 
  """
  sol_rad = solar_radiation_from_temp(tmin, tmax, date, lat, krs)
  net_sw_rad = net_in_sol_rad(sol_rad, albedo)
  net_lw_rad = net_out_lw_rad(tmin, tmax, h, date, lat, altitude, krs)
  
  return net_sw_rad - net_lw_rad

# ===============
# Potential Evapotranspiration
# ===============

def sat_vapour_pressure(t):
  """ Compute saturation vapour pressure [kPa].
      Based on equation 11 in Allen et al (1998).
      When using tdew, it calculates the actual vapour pressure.
      For daily mean: use with tmin and tmax: 1/2 * (sat_vapour_pressure(tmin) + sat_vapour_pressure(tmax))
  
  Parameters
  ----------
  t: array
    Temperature [C].

  Returns
  -------
  array
    saturation vapour pressure [kPa].
  """
  return 0.6108 * np.exp(17.27 * t / (t + 237.3))

def slope_sat_vapour_pressure(t):
  """ Compute slope of saturation vapour pressure [kPa/C].
      Based on equation 13 in Allen et al (1998).
  
  Parameters
  ----------
  t: array
    Temperature [C].
  
  Returns
  -------
  array
    Slope of saturation vapour pressure [kPa/C].
  """
  es = sat_vapour_pressure(t)
  return 4098.0 * es / np.power(t + 237.3, 2)

def actual_vapour_pressure(t, h):
  """ Compute actual vapour pressure [kPa]. 
      Based on equation 15 in Allen et al (1998).
  
  Parameters
  ----------
  t: array
    Temperature [C].
  h : array
    relative humidity [%].

  Returns
  -------
  array
    Actual vapour pressure [kPa].
  """
  es = sat_vapour_pressure(t)
  return es * h / 100

  ## alternative:
  #tdew = tdewpoint_from_relhum(t, h)
  #return sat_vapour_pressure(tdew)

def vapor_pressure_deficit(t, h):
  """ Compute vapour pressure deficite [kPa]. 
      Based on equation 15 in Allen et al (1998).
  
  Parameters
  ----------
  t: array
    Temperature [C].
  h : array
    relative humidity [%].

  Returns
  -------
  array
    Vapour pressure deficite [kPa].
  """
  es = sat_vapour_pressure(t)
  ea = actual_vapour_pressure(t, h)
  return es - ea
  # es - ea = es*(1-h/100)

def daylighthour_FAO(date, lat):
  """Compute daylight hours as maximum possible duration of sunshine for a given day of the year.

  Parameters
  ----------
  date: array
    Date in the format 'YYYYMMDD'.
  lat : array
    Latitude in degrees.

  Returns
  -------
  array
    daylight hours (duration of sunlight in hours).
  """

  # get Julian date: number of the day in the year between 1 (1 January) and 365 or 366 (31 December).
  day_of_year = datetime.datetime.strptime(date,'%Y%m%d').timetuple().tm_yday
  days_in_year = datetime.date(int(datetime.datetime.strptime(date,'%Y%m%d').year),12,31).timetuple().tm_yday
  
  # solar declination δ
  delta = 0.409 * np.sin((2 * np.pi * day_of_year / days_in_year)-1.39)
  # convert degrees to radians
  phi = np.deg2rad(lat)
  # sunset hour angle (rad)
  ws = np.arccos(-1 * np.tan(phi) * np.tan(delta))

  return ws * (24 / np.pi)

def heat_index(t_month):
  """Compute Heat Index I (for PET Thorntwaite)

  Parameters
  ----------
  t_month: array
    Monthly mean temperature [C]. Sequence of 12 values (for each month of the year from Jan - Dez).

  Returns
  -------
  scalar
    Heat Index.
  """

  # Negative temperatures should be set to zero
  adj_t_month = [t * (t >= 0) for t in t_month]

  # sum over months
  I = 0.0
  for t in adj_t_month:
    if t / 5.0 > 0.0:
      I += (t / 5.0) ** 1.514

def pot_evapo_thorntwaite(t, daylighthour, I):
  """Compute potential evapotranspiration after Thorntwaite [mm/day].

  Parameters
  ----------
  t: array
    Noon temperature [C].
  daylighthour : array
    Number of daylight hours according to FAO.
  I: scalar 
    Heat Index

  Returns
  -------
  array
    potential evapotranspiration after Thorntwaite [mm/day].
  """

  # exponent a
  a = 6.75 * (10**-7) * (I**3) - 7.71 * (10**-5) * (I**2) + 0.0179 * I + 0.492

  if t <= 26:
    return 16.0 * daylighthour / 360.0 * np.power(10.0 * max(0.0,t) / I, a)
  else: 
    return daylighthour / 360.0 * (-415.85 + 32.24 * t - 0.43 * (t**2))

def pot_evapo_penman(t, w, h, net_rad, altitude):
  """Compute potential evapotranspiration after Penman, after Shuttleworth [mm/day].

  Parameters
  ----------
  t : array
    temperature at 14:00 [C].
  w : array
    wind speed at 14:00 [m/s].
  h : array 
    relative humidity at 14:00 [%].
  net_rad : array 
    daily net radiation value [MJ·m-2·d-1].
  altitude : scalar
    altitude of location.

  Returns
  -------
  array
    potential evapotranspiration after Penman [mm/day].
  """

  # atmospheric pressure
  pa = 101.3 * np.power((293 - 0.0065 * altitude) / 293, 5.26)
  # latent heat of vaporization (Shuttleworth 1993)
  lhv = 2.501 - 0.002368 * t 
  # psychrometric constant (Shuttleworth 1993) [kPa/°C]
  gamma = 0.0016286 * pa / lhv   
  # slope of the saturation vapor pressure curve (Allen et al. 1998) [kPa/Â°C]      
  delta = slope_sat_vapour_pressure(t)
  # vapor pressure deficit value [kPa]
  vpd = vapor_pressure_deficit(t, h)

  return delta / (delta + gamma) * net_rad / lhv + gamma / (delta + gamma) * (6.43 * (1 + 0.536 * w) * vpd) / lhv

# ===============
# Angstroem index
# ===============

def angstroem(t, h):
  """Compute the Angstroem index.

  Parameters
  ----------
  t : array
    Noon temperature [C].
  h : array
    Noon relative humidity [%].
  
  Returns
  -------
  array
    Angstoem index value at current timestep.

  Example
  -------
  angstroem(17,42) == 3.1
  """
  return (h / 20) + ((27 - t) / 10)

# =================
# Baumgartner index
# =================

def baumgarner(pot_evapo_penman_fivedays, prec_fivedays):
  """Compute the cumulative Baumgartner index.

  Parameters
  ----------
  pot_evapo_penman_fivedays : array
    cumulative sum of five previous days potential evapotranspiration after Penman [mm/5day].
  prec_fivedays : array
    cumulative sum of five previous days rainfall [mm/5day].
  
  Returns
  -------
  array
    Baumgarner index value at current timestep.
  """
  return pot_evapo_penman_fivedays - prec_fivedays

def baumgartner_rating(baumgartner_index):
  """ Compute Baumgartner Danger Rating.

  Parameters
  ----------
  baumgartner_index : array
    Baumgarner index value.
  
  Returns
  -------
  class
    Baumgarner index value at current timestep.
  """

  # TO DO
  return baumgartner_index


# =================
# Nesterov index
# =================

def nesterov(t, h, p, prev):
  """Compute the cumulative Nesterov index.

  Parameters
  ----------
  t : array
    temperature at 15:00 [C].
  h : array
    relative humidity at 15:00 [%].
  p : array
    rainfall amount in open over previous 24 hours [mm].
  prev : array
    Previous day value of the Nesterov.
    prev_start = 0.0
  
  Returns
  -------
  array
    Nesterov index value at current timestep.

  Example
  -------
  nesterov(20,50,10,0) == 214.598280
  """

  tdew = tdewpoint_from_relhum(t, h)
  p_threshold = 3 # daily precipitation (p) does not exceed 3 mm 
  if p > p_threshold:
    return 0.0
  else:
    return prev + (t - tdew) * t

# =================
# Munger Drought index
# =================

def munger(p, prev):
  """Compute the cumulative Munger Drought index.

  Parameters
  ----------
  p : array
    rainfall amount in open over previous 24 hours [mm].
  prev : array
    Previous day value of the Munger.
    prev_start = 0.0
  
  Returns
  -------
  array
    Munger index value at current timestep.

  Example
  -------
  munger(1,0.5) == 2
  """

  p_threshold = 1.27 # daily precipitation (p) does not exceed 1.27 mm (0.05 inch)
  if p >= p_threshold:
    return 0.0
  else:
    return 0.5 * np.power(np.sqrt(2 * prev) + 1, 2)

# ===============================================
# Fuel moisture index (FMI) and Forest fire danger rating index (F)
# ===============================================

def fuel_moisture_index(t, h):
  """Compute the Fuel moisture index after Sharples et al. (2009a, 2009b).

  Parameters
  ----------
  t : array
    temperature [C].
  h : array
    relative humidity [%].
  
  Returns
  -------
  array
    Fuel moisture index value at current timestep.

  Example
  -------

  """
  return 10 - 0.25 * (t - h)

def fmi_danger_rating(t, h, w):
  """Compute the Forest fire danger rating index (F) after Sharples et al. (2009a, 2009b).

  Parameters
  ----------
  t : array
    temperature [C].
  h : array
    relative humidity [%].
  w : array
    wind speed [m/s].
  
  Returns
  -------
  array
    Fuel moisture index value at current timestep.

  Example
  -------

  """
  windthreshold = 1
  fmi = fuel_moisture_index(t, h)
  return max(windthreshold, w * 3.6) / fmi


# Modification with drought factor from McArthur Mark 5 forest fire danger index 

# ===============================================
# Fosberg fire weather index 
# ===============================================

# requires hourly observations of temperature, relative air humidity and wind speed as input data

# ===============================================
# Keetch-Byram drought index 
# ===============================================

def kbdi(tmax, p, kbdi0, psum0, pweek, pWeekThreshold, pAnnualAvg):
  """ Compute the Keetch-Byram drought index in S.I. units from Crane (1982).
  It represents the drying (i.e. the increase moisture deficency in mm) due to temperature (evapotranspiration)
  for a given location. The assumprion is that the mean annual precipitation is used as a proxy
  for the amount of vegetation present.
  
  Parameters
  ----------
  tmax : array
    Daily maximal temperature [°C].
  p : array
    Rainfall amount [mm].
  kbdi0 : array
    previous day KBDI Value.
  pWeekThreshold : array 
    weekly rain threshold to initialize index [mm]. Often 30mm is used.
  pAnnualAvg : array
    Annual rainfall average [mm].
  psum0 : array
    sum of consecutive rainfall [mm].
  pweek : array
    sum of rain over last 7 days [mm].
  
  Returns
  -------
  array
    Keetch-Byram drought index.

  Example
  -------
  kbdi(tmax=27,p=9,kbdi0=200,pAnnualAvg=200) == 196
  """
  deltatime = 1
  pthreshold = 0.2 * 25.4 #  = 5.08 mm (convert inches to mm)
  pnet = max(0, p - max(0, pthreshold - (psum0 - p)))

  if (pweek >= pWeekThreshold):
    Q = 0.0
  else:
    Q = max(0, kbdi0 - pnet) # yesterday KBDI minus effective precipitation
  
  # Potential Evapotranspiration
  numerator = (203.2 - Q) * (0.968 * np.exp(0.0875 * tmax + 1.5552) - 8.30) * deltatime
  denominator = 1 + 10.88 * np.exp(-0.001736 * pAnnualAvg)
  
  return Q + max(0, numerator / denominator * 0.001)

def rain_sum(p, prev):
  """ Compute amount (sum) of consecutive rainfall

  Parameters
  ----------
  p : array
    rainfall amount in open over previous 24 hours [mm].
  prev : array
    Previous sum of consecutive rain.
    prev_start = 0.0

  Returns
  -------
  array
    sum of consecutive rainfall [mm]
  """
  if p == 0:
    return 0
  else:
    return prev + p


def days_since_rain(p, prev, threshold=0):
  """Compute consecutive number of days without/little rain.

  Parameters
  ----------
  p : array
    rainfall amount in open over previous 24 hours [mm].
  prev : array
    Previous number of consecutive days without rain.
    prev_start = 0.0
  threshold: scalar 
    Threshold for dry days

  Returns
  -------
  array
    Number of days without rain at current timestep.
  """
  if p > threshold:
    return 0
  else:
    return prev + 1

# requires daily temperature, daily and annual precipitation as input data.

# ===============================================
# McArthur Mark 5 forest fire danger index
# ===============================================

# requires temperature, relative humidity, wind speed and a fuel availability index (i.e. a drought factor) measured at 15:00 as input variables

# ===============================================
# Canadian Forest Fire Weather Index System (FWI)
# ===============================================

# FWI is initialized with some values for FFMC, DMC and DC components. This means that the first values of the series are not reliable,
# until the index is iterated over several time steps and stabilizes (typically a few days suffice).

# Reference: Wang, Anderson and Suddaby, 2015.
day_lengths = np.array(
  [
      [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8],
      [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
      12 * [9],
      [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
      [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6],
  ]
)

drying_factors = np.array(
  [
      [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
      12 * [1.39],
      [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
  ]
)

@jit
#@vectorize
def _day_length(lat: Union[int, float], mth: int):  # Union[int, float]
  """Return the average day length for a month within latitudinal bounds."""
  if -30 > lat >= -90:
      dl = day_lengths[0, :]
  elif -15 > lat >= -30:
      dl = day_lengths[1, :]
  elif 15 > lat >= -15:
      return 9
  elif 30 > lat >= 15:
      dl = day_lengths[3, :]
  elif 90 >= lat >= 30:
      dl = day_lengths[4, :]
  elif lat > 90 or lat < -90:
      raise ValueError("Invalid lat specified.")
  else:
      raise ValueError
  return dl[mth - 1]

@jit
def _drying_factor(lat: float, mth: int):  
  """Return the day length factor / drying factor."""
  if -15 > lat >= -90:
      dlf = drying_factors[0, :]
  elif 15 > lat >= -15:
      return 1.39
  elif 90 >= lat >= 15:
      dlf = drying_factors[2, :]
  elif lat > 90 or lat < -90:
      raise ValueError("Invalid lat specified.")
  else:
      raise ValueError
  return dlf[mth - 1]


@vectorize
def ffmc(t, p, w, h, ffmc0):
  """Compute the fine fuel moisture code over one timestep (canadian index).

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
  ffmc0 : array
    Previous value of the fine fuel moisture code.
    ffmc_start = 85.0.
  
  Returns
  -------
  array
    Fine fuel moisture code at the current timestep.

  Example
  -------
  ffmc(17,0,6.944,42,85) == 87.69298009277445

  To Do
  -----
  - if ffmc0 is None, then ffmc0 = ffmc0_start
  - snowcover as parameter (0/1). If snowcover == 1, return ffmc = Nan
  """
  # clip humidity to [0,100]
  #h = np.clip(h, 0, 100) # does not work with @vectorize
  #if ffmc0 is None:
  #    ffmc0 = 85.0

  # convert wind speed from m/s to km/h
  w = w * 3.6

  # Eq. 1
  mo = (147.2 * (101.0 - ffmc0)) / (59.5 + ffmc0) 
  
  if p > 0.5:
      rf = p - 0.5  # Eq.2: Rain reduction to allow for loss in overhead canopy ("effective rainfall")
      if mo > 150.0:
          mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
          ) + (0.0015 * (mo - 150.0) ** 2) * np.sqrt(rf)
          # Eq.3b
      elif mo <= 150.0:
          mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
          # Eq.3a: The real moisture content of pine litter ranges up to about 250 percent, so we cap it at 250
      if mo > 250.0:
          mo = 250.0
  # Eq.4: Equilibrium moisture content from drying phase
  ed = (
      0.942 * (h ** 0.679)
      + (11.0 * np.exp((h - 100.0) / 10.0))
      + 0.18 * (21.1 - t) * (1.0 - 1.0 / np.exp(0.1150 * h))
  ) 

  if mo < ed:
        # Eq. 5 Equilibrium moisture content from wetting phase
      ew = (
          0.618 * (h ** 0.753)
          + (10.0 * np.exp((h - 100.0) / 10.0))
          + 0.18 * (21.1 - t) * (1.0 - 1.0 / np.exp(0.115 * h))
      )  
      if mo < ew:
          #Eq. 7a (ko) Log wetting rate at the normal temperature of 21.1 C 
          kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + (
              0.0694 * np.sqrt(w)
          ) * (1.0 - ((100.0 - h) / 100.0) ** 8)
          # Eq. 7b Affect of temperature on wetting rate
          kw = kl * (0.581 * np.exp(0.0365 * t))  
          # Eq. 9
          m = ew - (ew - mo) / 10.0 ** kw  
      elif mo > ew:
          m = mo
  elif mo == ed:
      m = mo
  else: # if mo > ed
        #Eq. 6a (ko) Log drying rate at the normal temperature of 21.1 C
      kl = 0.424 * (1.0 - (h / 100.0) ** 1.7) + (0.0694 * np.sqrt(w)) * (
          1.0 - (h / 100.0) ** 8
      )
      # Eq. 6b Affect of temperature on  drying rate
      kw = kl * (0.581 * np.exp(0.0365 * t))
      # Eq.8
      m = ed + (mo - ed) / 10.0 ** kw

  # Eq. 10 Final ffmc calculation
  ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
  # Constraints: ffmc is scaled between 0 and 101
  # ffmc = min(max(0.0,ffmc),101.0)
  if ffmc > 101.0:
      ffmc = 101.0
  elif ffmc <= 0.0:
      ffmc = 0.0
  
  return ffmc

@vectorize
def dmc(t, p, h, mth: int, lat: float, dmc0: float): 
  """Compute the Duff moisture code over one time step (canadian index).

  Parameters
  ----------
  t: array
    Noon temperature [C].
  p : array
    rainfall amount in open over previous 24 hours, at noon [mm].
  h : array
    Noon relative humidity [%].
  mth : integer array
    Month of the year [1-12].
  lat : float
    Latitude in degrees.
  dmc0 : float
    Previous value of the Duff moisture code.

  Returns
  -------
  array
    Duff moisture code at the current timestep

  Example
  ------- 
  dmc(17,0,42,6,45.98,6) == 8.5450511359999997
  """
  # clip humidity to [0,100]
  #h = np.clip(h, 0, 100)
  
  #if dmc0 is None:
  #    dmc0 = 6

  if t < -1.1:
      rk = 0
  else:
      dl = _day_length(lat, mth)
      # Eqs.16 and 17
      rk = 1.894 * (t + 1.1) * (100.0 - h) * dl * 0.0001  

  if p > 1.5:
      ra = p
      # Eq.11 Effective rainfall
      rw = 0.92 * ra - 1.27  
      # Eq.12 from R-package cffdrs
      wmi = 20.0 + 280.0 / np.exp(0.023 * dmc0) 
      if dmc0 <= 33.0:
          # Eq.13a
          b = 100.0 / (0.5 + 0.3 * dmc0)  
      else: # dmc0 > 33.0
          if dmc0 <= 65.0:
              # Eq.13b
              b = 14.0 - 1.3 * np.log(dmc0)  
          else:
                # Eq.13c
              b = 6.2 * np.log(dmc0) - 17.2 
      # Eq.14 duff moisture content after p
      wmr = wmi + (1000 * rw) / (48.77 + b * rw)  
      # Eq.15
      pr = 43.43 * (5.6348 - np.log(wmr - 20.0))  
  else:  # p <= 1.5
      pr = dmc0
  
  if pr < 0.0:
      pr = 0.0
  # Calculate final dmc
  dmc = pr + rk
  # Constraints: dmc is scaled between max(0, dmc)
  if dmc < 0:
      dmc = 0.0
      
  return dmc

@vectorize
def dc(t, p, mth, lat, dc0):  
  """Compute the drought code over one time step (canadian index).

  Parameters
  ----------
  t: array
    Noon temperature [C].
  p : array
    rainfall amount in open over previous 24 hours, at noon [mm].
  mth : integer array
    Month of the year [1-12].
  lat : float
    Latitude.
  dc0 : float
    Previous value of the drought code.

  Returns
  -------
  array
    Drought code at the current timestep

  Example
  ------- 
  dc(17,0,4,45.98,15) == 19.013999999999999
  """
  fl = _drying_factor(lat, mth) # influence of latitude, from R-package cffdrs

  if t < -2.8:
      t = -2.8
  # Eq.22 Potential Evapotranspiration
  pe = (0.36 * (t + 2.8) + fl) / 2  
  if pe < 0.0:
      pe = 0.0

  if p > 2.8:
      ra = p
      # Eq.18 Effective rainfall
      rw = 0.83 * ra - 1.27  
      # Eq.19 Moisture equivalent of the previous day's DC
      smi = 800.0 * np.exp(-dc0 / 400.0)  
      # Eqs.20 and 21
      dr = dc0 - 400.0 * np.log(1.0 + ((3.937 * rw) / smi))  
      if dr > 0.0:
          dc = dr + pe
      elif np.isnan(dc0):
          dc = np.NaN
      else:
          dc = pe
  else:  # if precip is less than 2.8 then use yesterday's DC
      dc = dc0 + pe
  return dc

def isi(w, ffmc):
  """Initialize spread index (canadian index).

  Parameters
  ----------
  w : array
    Noon wind speed [m/s].
  ffmc : array
    Fine fuel moisture code.

  Returns
  -------
  array
    Initial spread index.

  Example
  ------- 
  isi(6.944444444444445,87.6929800927744) == 10.853661073655068
  """
  # convert wind speed from m/s to km/h
  w = w * 3.6
  # Eq.1  Moisture content
  mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)  
  # Eq.25 Fine Fuel Moisture
  ff = 19.1152 * np.exp(mo * -0.1386) * (1.0 + (mo ** 5.31) / 49300000.0)  
  # Eq.26 Spread Index Equation (with Wind Effect)
  isi = ff * np.exp(0.05039 * w)  
  return isi

def bui(dmc, dc):
  """Build-up index (canadian index).

  Parameters
  ----------
  dmc : array
    Duff moisture code.
  dc : array
    Drought code.

  Returns
  -------
  array
    Build up index.
  
  Example
  ------- 
  bui(8.5450511359999997,19.013999999999999) == 8.4904265358371838
  """
  bui = np.where(
      dmc <= 0.4 * dc,
      # Eq.27a
      (0.8 * dc * dmc) / (dmc + 0.4 * dc),  
      # Eq.27b
      dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7),
  )  
  return np.clip(bui, 0, None)


def fwi(isi, bui):
  """Fire weather index in S-scale (canadian index).

  Parameters
  ----------
  isi : array
    Initial spread index
  bui : array
    Build up index.

  Returns
  -------
  array
    Fire weather index.
  
  Example
  ------- 
  fwi(10.853661073655068,8.4904265358371838) = 10.096371392382368
  """
  fwi = np.where(
      bui <= 80.0,
      # Eq.28a
      0.1 * isi * (0.626 * bui ** 0.809 + 2.0),  
      # Eq.28b
      0.1 * isi * (1000.0 / (25.0 + 108.64 / np.exp(0.023 * bui))),
  )  
  # Eqs.30a and 30b Constraint if fwi > 1
  fwi[fwi > 1] = np.exp(2.72 * (0.434 * np.log(fwi[fwi > 1])) ** 0.647)  
  return fwi

def daily_severity_rating(fwi):
  """Daily severity rating (canadian index).

  Parameters
  ----------
  fwi : array
    Fire weather index

  Returns
  -------
  array
    Daily severity rating.
  """
  return 0.0272 * fwi ** 1.77

