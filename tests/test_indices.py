# run in project folder (not in /tests folder)

# import packages
import pytest
import warnings
import pandas as pd
import numpy as np
import firedanger.indices as indices

# to do
# - test with array / xarray

# define functions
def nearlyEqual(a, b):
    '''useful for comparing floats that may not be identical but are close enough'''
    if abs(a-b) < 5*pow(10,-5):
        return True
    return False

def add(x,y):
    return x+y

# tests
def test_add():
    assert add(1,2) == 3
    assert nearlyEqual(add(1,2),3)

# Meteorological variables
def test_meteo():
    # dewpoint temperature
    assert nearlyEqual(indices.tdewpoint_from_relhum(20,50),9.270085985370075)
    # days since rain
    prev = 0
    for precip, result in zip(np.array((0,0,0,1,0,0)), np.array((1,2,3,0,1,2))):
        assert nearlyEqual(indices.days_since_rain(precip,prev),result)
        prev = indices.days_since_rain(precip,prev)

# Angström

def test_angstroem():
    assert nearlyEqual(indices.angstroem(17,42),3.1)

# Nesterov

def test_nesterov():
    assert nearlyEqual(indices.nesterov(20,50,10,100),0) # p > p_threshold of 3 mm
    assert nearlyEqual(indices.nesterov(20,50,0,0),214.598280) # p < p_threshold of 3 mm

# Canadian FWI

def test_ffmc():
    assert nearlyEqual(indices.ffmc(17,0,6.944,42,85),87.69298009277445)
    # humidity h is capped at 100
    #assert nearlyEqual(indices.ffmc(17,0,6.944,103,85),75.04092243101171)

def test_dmc():
    assert nearlyEqual(indices.dmc(17,0,42,4,45.98,6), 8.5450511359999997)

def test_dc():
    assert nearlyEqual(indices.dc(17,0,4,45.98,15),19.013999999999999)

def test_isi():
    assert nearlyEqual(indices.isi(6.944444444444445,87.6929800927744),10.853661073655068)

def test_bui():
    assert nearlyEqual(indices.bui(8.5450511359999997,19.013999999999999),8.4904265358371838)

def test_fwi():
    assert nearlyEqual(indices.fwi(10.853661073655068,8.4904265358371838),10.096371392382368)

def test_dsr():
    assert nearlyEqual(indices.daily_severity_rating(10.096371392382368),1.6290766399790664)

def test_time_series():
    # read csv file (this is coming from the R package cffdrs)
    df_test = pd.read_csv("tests/test_fwi.csv", sep=',', index_col=0)
    for index, row in df_test.iloc[1:].iterrows():
        #print(index, row['TEMP'],row['PREC'])
        assert nearlyEqual(indices.ffmc(row['TEMP'],row['PREC'],row['WS']/3.6,row['RH'],df_test.loc[index-1]['FFMC']),row['FFMC'])
        assert nearlyEqual(indices.dmc(row['TEMP'],row['PREC'],row['RH'],row['MON'].astype(int),row['LAT'],df_test.loc[index-1]['DMC']),row['DMC'])
        assert nearlyEqual(indices.dc(row['TEMP'],row['PREC'],row['MON'].astype(int),row['LAT'],df_test.loc[index-1]['DC']),row['DC'])
        assert nearlyEqual(indices.isi(row['WS']/3.6,row['FFMC']),row['ISI'])
        assert nearlyEqual(indices.bui(row['DMC'],row['DC']),row['BUI'])
        assert nearlyEqual(indices.fwi(row['ISI'],row['BUI']),row['FWI'])
        assert nearlyEqual(indices.daily_severity_rating(row['FWI']),row['DSR'])