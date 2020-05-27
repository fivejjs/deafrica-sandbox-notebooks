# deafrica_phenology.py

import sys
import numpy as np
import xarray as xr
from scipy.stats import skew
sys.path.append('../Scripts')
from deafrica_datahandling import first, last

def _vpos(da):
    """
    vPOS = Value at peak of season
    """
    return da.max('time')

def _pos(da):
    """
    POS = DOY of peak of season
    """
    return da.isel(time=da.argmax('time')).time.dt.dayofyear

def _trough(da):
    """
    Trough = Minimum value in the timeseries
    """
    return da.min('time')

def _aos(vpos, trough):
    """
    AOS = Amplitude of season
    """
    return vpos - trough

def _vsos(da, pos, method_sos='first'):
    """
    vSOS = Value at the start of season
    
    method : If 'first' then SOS is estimated
            as the first positive slope on the
            greening side of the curve. If median,
            then SOS is estimated as the median value
            of the postive slopes on the greening side of the
            curve.
    """
    # select timesteps before peak of season (AKA greening)
    greenup = da.where(da.time < pos.time)
    # find the first order slopes
    green_deriv = greenup.differentiate('time')
    # find where the fst order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv>0)
    
    if method_sos=='first':  
        # get the timestep where slope first becomes positive to estimate
        # the DOY when growing season starts
        return first(pos_green_deriv, dim='time')
    
    if method_sos == 'median':
        #grab only the increasing greening values
        pos_greenup = greenup.where(pos_green_deriv)
        # find the median "actual" value of those positive greening values
        # (rather than the mean of the two median values for even-numbered lengths)
        median_val = pos_greenup.quantile(0.5, dim='time', interpolation='nearest', skipna=True)
        # find the locations that match the median value 
        return first(pos_greenup.where(pos_greenup==median_val), dim='time')

def _sos(vsos):
    """
    SOS = DOY of start of season
    """
    return vsos.time.dt.dayofyear

def _veos(da, pos, method_eos='last'):
    """
    vEOS = Value at the start of season
    
    method : If 'first' then EOS is estimated
            as the last negative slope on the
            senescing side of the curve. If median,
            then EOS is estimated as the median value
            of the negative slopes on the senescing 
            side of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    senesce = da.where(da.time > pos.time)
    # find the first order slopes
    senesce_deriv = senesce.differentiate('time')
    # find where the fst order slope is postive
    neg_senesce_deriv = senesce_deriv.where(senesce_deriv < 0)
    
    if method_eos == 'last':  
        # get the timestep where slope first becomes positive to estimate
        # the DOY when growing season starts
        return last(neg_senesce_deriv, dim='time')
    
    if method_eos == 'median':
        #grab only the declining values
        neg_greenup = senesce.where(neg_senesce_deriv)
        # find the median "actual" value of those positive greening values
        # (rather than the mean of the two median values for even-numbered lengths)
        median_val = neg_greenup.quantile(0.5, dim='time', interpolation='nearest', skipna=True)
        # find the locations that match the median value 
        return last(neg_greenup.where(neg_greenup==median_val), dim='time')

def _eos(veos):
    """
    EOS = DOY of end of seasonn
    """
    return veos.time.dt.dayofyear

def _los(eos, sos):
    """
    LOS = Length of season (DOY)
    """
    return eos - sos

def _rog(vpos, vsos, pos, sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos - vsos) / (sos - pos)


def _ros(veos, vpos, eos, pos):
    """
    ROG = Rate of Senescing (Days)
    """
    return (veos - vpos) / (eos - pos)


def xr_phenology(da,
                 stats=[
                     'SOS', 'POS', 'EOS', 'Trough'
                     'vSOS', 'vPOS', 'vEOS', 'LOS',
                     'AOS', 'ROG', 'ROS'],
                 method_sos='first',
                 method_eos='last',
                 interpolate=False,
                 interpolate_na=False,
                 interp_method="linear",
                 interp_interval='1W'):
    
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
        the metrics returned, all statistics are calculated
        due to inter-dependencies between metrics.
        Options include:
            SOS = DOY of start of season
            POS = DOY of peak of season
            EOS = DOY of end of season
            vSOS = Value at start of season
            vPOS = Value at peak of season
            vEOS = Value at end of season
            Trough = Minimum value of season
            LOS = Length of season (DOY)
            AOS = Amplitude of season (in value units)
            ROG = Rate of greening
            ROS = Rate of senescence
            Skew = Skewness of growing season (NOT IMPLEMENTED YET)
            IOS = Integral of season (SOS-EOS) (NOT IMPLEMENTED YET)

    Outputs
    -------
        xarray.Dataset containing variables for the selected statistics 
        
    """
    if (interpolate_na==True) & (interpolate==True):
        da = da.interpolate_na(dim='time', method=interp_method) 
        print('removed NaNs')
        
        #resample time dim and interpolate values
        da=da.resample(time=interp_interval).interpolate(interp_method)
        print("    Interpolated dataset to " +str(len(da.time))+ " time-steps")
     
    if (interpolate_na==False) & (interpolate==True):
        da=da.resample(time=interp_interval).interpolate(interp_method)
        print("Interpolated dataset to " +str(len(da.time))+ " time-steps")
        
    if (interpolate_na==True) & (interpolate==False):
        da = da.interpolate_na(dim='time', method=interp_method)
        print('removed NaNs')
   
    vpos = _vpos(da)
    pos = _pos(da)
    trough = _trough(da)
    aos = _aos(vpos, trough)
    vsos = _vsos(da, pos, method_sos=method_sos)
    sos = _sos(vsos)
    veos = _veos(da, pos, method_eos=method_eos)
    eos = _eos(veos)
    los = _los(eos, sos)
    rog = _rog(vpos, vsos, pos, sos)
    ros = _ros(veos, vpos, eos, pos)

    # Dictionary containing the statistics
    stats_dict = {
        'SOS': sos,
        'EOS': eos,
        'vSOS': vsos,
        'vPOS': vpos,
        'Trough': trough,
        'POS': pos,
        'vEOS': veos,
        'LOS': los,
        'AOS': aos,
        'ROG': rog,
        'ROS': ros,
    }

    #intialise dataset with first statistic
    ds = stats_dict[stats[0]].to_dataset(name=stats[0])

    #add the other stats to the dataset
    for stat in stats[1:]:
        stats_keep = stats_dict.get(stat)
        ds[stat] = stats_dict[stat]

    return ds


#STATS TO BE IMPLEMENTED
# def _ios(da, sos, eos, dt_unit='D'):
#     """
#     IOS = Integral of season (SOS-EOS)
        
#     dt_unit : str, optional
#             Can be used to specify the unit if datetime
#             coordinate is used. 
#             One of {‘Y’, ‘M’, ‘W’, ‘D’, ‘h’, ‘m’,
#             ‘s’, ‘ms’, ‘us’, ‘ns’, ‘ps’, ‘fs’, ‘as’}
#     """
#     season = da.where((ndvi.time > sos.time) & (ndvi.time < eos.time))
#     return season.integrate(dim='time', datetime_unit=dt_unit)

# def _skew(da, sos, eos):
#     """
#     skew= skewness of growing season (SOS to EOS)
#     """
#     season = da.where((ndvi.time > sos.time) & (ndvi.time < eos.time))
#     return xr.apply_ufunc(skew,
#                           season,
#                           input_core_dims=[["time"]],
#                           dask='allowed')