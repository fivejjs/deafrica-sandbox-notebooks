# deafrica_phenology.py



import numpy as np
import xarray as xr
from scipy.stats import skew

def _vpos(da):
    """
    vPOS = Value at peak of season
    """
    return da.max('time')

def _ipos(da):
    """
    IPOS = time index for peak of season
    """
    return da.isel(time=da.argmax('time')).time

def _pos(da):
    """
    POS = DOY of peak of season
    """
    return da.isel(time=da.argmax('time')).time.dt.dayofyear

def _trough(da):
    return da.min('time')

def _aos(vpos, trough):
    """
    AOS = Amplitude of season
    """
    return vpos - trough

def _sos(da, ipos, method_sos='first'):
    """
    SOS = DOY of start of season
    
    method : If 'first' then SOS is estimated
            as the first positive slope on the
            greening side of the curve. If median,
            then SOS is estimated as the median value
            of the postive slopes on the greening side of the
            curve.
    """
    #select timesteps before peak of season (AKA greening)
    greenup=da.sel(time=slice(da.time.values[0], ipos.values))
    # find the first order slopes
    green_deriv = greenup.differentiate('time')
    # find where the fst order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv>0, drop=True)
    
    if method_sos=='first':  
        # get the timestep where slope first becomes positive to estimate
        # the DOY when growing season starts
        return pos_green_deriv[0].time.dt.dayofyear
    
    if method_sos == 'median':
        #grab only the increasing greening values
        pos_greenup = greenup.where(pos_green_deriv, drop=True)
        #calulate the median of those positive greening values
        median = pos_greenup.median('time')
        # To determine 'time-of' the median calculate the distance
        # each value has from median
        distance = pos_greenup - median
        #determine location of the value with the minimum distance from
        # the median
        idx = distance.where(distance==distance.min(), drop=True)
        return idx.time.dt.dayofyear
    
def _vsos(da, sos):
    """
    vSOS = Value at the start of season
    """
    return da.sel(time=sos.time)

def _eos(da, ipos, method_eos='last'):
    """
    EOS = DOY of end of season
    
    method : If 'first' then EOS is estimated
            as the last negative slope on the
            senescing side of the curve. If median,
            then EOS is estimated as the median value
            of the negative slopes on the senescing 
            side of the curve.
    """
    #select timesteps after peak of season
    senesce=da.sel(time=slice(ipos.values, da.time[-1].values))
    # find the first order slopes
    senesce_deriv = senesce.differentiate('time')
    # find where the fst order slope is negative
    neg_senesce_deriv = senesce_deriv.where(senesce_deriv<0, drop=True)
    
    if method_eos=='last':  
        # get the timestep where slope first becomes positive to estimate
        # the DOY when growing season starts
        return neg_senesce_deriv[-1].time.dt.dayofyear
    
    if method_eos == 'median':
        #grab only the declining values
        neg_greenup = senesce.where(neg_senesce_deriv, drop=True)
        #calulate the median of those positive greening values
        median = neg_greenup.median('time')
        # To determine 'time-of' the median, calculate the distance
        # each value has from median
        distance = neg_greenup - median
        #determine location of the value with the minimum distance from
        #the median
        idx = distance.where(distance==distance.min(), drop=True)
        return idx.time.dt.dayofyear    

def _veos(da, eos):
    """
    vSOS = Value at the start of season
    """
    return da.sel(time=eos.time)

def _los(eos, sos):
    """
    LOS = Length of season (DOY)
    """
    return eos - sos

def _ios(da, sos, eos, dt_unit='D'):
    """
    IOS = Integral of season (SOS-EOS)
        
    dt_unit : str, optional
            Can be used to specify the unit if datetime
            coordinate is used. 
            One of {‘Y’, ‘M’, ‘W’, ‘D’, ‘h’, ‘m’,
            ‘s’, ‘ms’, ‘us’, ‘ns’, ‘ps’, ‘fs’, ‘as’}
    """
    season = da.sel(time=slice(sos.time, eos.time))
    return season.integrate(dim = 'time', datetime_unit=dt_unit)

def _rog(vpos,vsos,pos,sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos-vsos) / (pos - sos)
    
def _rog(vpos,vsos,pos,sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos-vsos) / (pos - sos) 

def _ros(veos,vpos,eos,pos):
    """
    ROG = Rate of Senescing (Days)
    """
    return (veos-vpos) / (eos - pos) 

def _skew(da, sos, eos):
    """
    skew= skewness of growing season (SOS to EOS)
    """
    season = season = da.sel(time=slice(sos.time, eos.time))
    return xr.apply_ufunc(
            skew,
            season,
            input_core_dims=[["time"]],
            dask='allowed')
    

def xr_phenology(da,
                 stats=[
                     'SOS',
                     'EOS',
                     'vSOS',
                     'vPOS',
                     'vEOS',
                     'LOS',
                     'AOS',
                     'IOS',
                     'ROG',
                     'ROS',
                     'skew'],
                    method_sos='first',
                    method_eos='last',
                    dt_unit='D'):
    
    """
    Obtain land surface phenology metrics from an
    xarray.DataArray containing a timeseries of a remote-sensinh
    vegetation index e.g. NDVI or EVI.
    
    last modified May 2020
    
    Parameters
    ----------
    da :  xarray.Dataset
    stats : list
        list of phenological statistics to return. Regardless of
        the metrics returned, all statistics are
        calculated due to inter-dependicises between metrics.
        Options include:
            POS = DOY of peak of season
            EOS = DOY of end of season
            vSOS = Value at start of season
            vPOS = Value at peak of season
            vEOS = Value at end of season
            LOS = Length of season (DOY)
            AOS = Amplitude of season (in value units)
            IOS = Integral of season (SOS-EOS)
            ROG = Rate of greening
            ROS = Rate of senescence
            SW = Skewness of growing season

    Outputs
    -------
        xarray.Dataset containing variables for the selected statistics 
        
    """
            
    vpos=_vpos(da)
    ipos = _ipos(da)
    pos = _pos(da)
    trough = _trough(da)
    aos = _aos(vpos, trough)
    sos = _sos(da, ipos, method_sos=method_sos)
    vsos = _vsos(da, sos)
    eos = _eos(da, ipos, method_eos=method_eos)
    veos = _veos(da, eos)
    los = _los(eos, sos)
    ios = _ios(da, sos, eos, dt_unit=dt_unit)
    rog = _rog(vpos,vsos,pos,sos)
    ros = _ros(veos,vpos,eos,pos)
    skew = _skew(da, sos, eos)
    
    # Dictionary containing the statistics
    stats_dict = {'SOS':sos,
                 'EOS':eos,
                 'vSOS':vsos,
                 'vPOS':vpos,
                 'POS':pos,
                 'vEOS':veos,
                 'LOS':los,
                 'AOS':aos,
                 'IOS':ios,
                 'ROG':rog,
                 'ROS':ros,
                 'Skew':skew}
    
    #intialise dataset with first statistic
    ds = stats_dict[stats[0]].to_dataset(name=stats[0])
    
    #add the other stats to the dataset
    for stat in stats[1:]:
        stats_keep = stats_dict.get(stat)
        ds[stat] = stats_dict[stat]

    return ds

