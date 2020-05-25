
import numpy as np
import xarray as xr
from scipy.integrate import trapz
from scipy.stats import skew

# fucntions to calculate each statistc
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

def _sos(da, ipos, method='first'):
    """
    SOS = DOY of start of season
    
    method : If 'first' then iSOS is estimated
            as the first positive slope on the
            greening side of the curve. If median,
            then iSOS is estimated as the median value
            of the postive slopes on the greening side of the
            curve.
    """
    #select timesteps before peak of season (AKA greening)
    greenup=da.sel(time=slice(da.time[0].values, ipos.values))
    # find the first order slopes
    green_deriv = greenup.differentiate('time')
    # find where the fst order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv>0, drop=True)
    
    if method=='first':  
        # get the timestep where slope first becomes positive to estimate
        # the DOY when growing season starts
        return pos_green_deriv[0].time.dt.dayofyear
    
    if method == 'median':
        #grab only the positive greening values
        pos_greenup = greenup.where(pos_green_deriv, drop=True)
        #calulate the median of those positive greening values
        median = pos_greenup.median('time')
        # To determine 'time-of' the median calculate the distance
        # each value has from median
        distance = pos_greenup - median
        #determine location of the value with the minimum distance from
        #the median
        idx = distance.where(distance==distance.min(), drop=True)
        return idx.time.dt.dayofyear
    
def _vsos(da, sos):
    """
    vSOS = Value at the start of season
    """
    return da.sel(time=sos.time)

    
    
# def _isos(doy, sos):
#     """
#     isos = index of start of season
#     """
#     return np.where(doy == int(sos))[0]

# def _eos(ratio, doy, sos):
#     """
#     EOS = DOY of end of season
#     """
#     # separate greening from senesence values
#     dev = np.gradient(ratio)  # first derivative
#     greenup = np.zeros([ratio.shape[0]],  dtype=bool)
#     greenup[dev > 0] = True
#     i = np.nanmedian(doy[ipos[0][0]:][~greenup[ipos[0][0]:]])
#     eos = doy[(np.abs(doy - i)).argmin()]
#     if eos is None:
#         ieos = len(doy) - 1
#         eos = doy[ieos]
#     return eos

# def _ieos(doy, eos):    
#     return np.where(doy == eos)[0]

# def _vsos(da, isos):
#     """
#     vSOS = Value at start of season
#     """
#     return da[isos][0]

# def _veos(da, ieos):
#     """
#     vEOS = Value at end of season
#     """
#     return da[ieos][0]

# def _los(eos, sos, da):
#     """
#     LOS = Length of season (DOY)
#     """
#     los = eos - sos
#     if los < 0:
#         los[los < 0] = len(da) + (eos[los < 0] - sos[los < 0])
#     return los

# def _rog(doy, sos, eos, da, vpos, isos, pos):
#     """
#     ROG = Rate of greening
#     """
#     green = doy[(doy > sos) & (doy < eos)]
#     _id = []
#     for i in range(len(green)):
#         _id.append((doy == green[i]).nonzero()[0])
#     _id = np.array([item for sublist in _id for item in sublist])
#     ios = trapz(da[_id], doy[_id])
#     rog = (vpos - da[isos]) / (pos - sos)
#     return rog[0]
    
# def _ros(da, ieos, vpos, eos, pos):
#     """
#      ROS = Rate of senescence
#     """
#     ros = (da[ieos] - vpos) / (eos - pos)
#     return ros[0]

# def _sw(da, doy, sos, eos):
#     """
#     SW = Skewness of growing season
#     """
#     green = doy[(doy > sos) & (doy < eos)]
#     _id = []
#     for i in range(len(green)):
#         _id.append((doy == green[i]).nonzero()[0])
#     _id = np.array([item for sublist in _id for item in sublist]) 
#     return skew(da[_id])

# def _vsos(da, isos):
#     """
#     vSOS = Value at start of season
#     """
#     return da[isos][0]

# def _veos(da, ieos): 
#     """
#     vEOS = Value at end of season
#     """
#     return da[ieos][0]


def xr_phenology(ds,
                 doy,
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
                     'SW'],
                 drop=True):
    
    """
    Obtain land surfurface phenology metrics from an
    xarray.Dataset that contains a a timeseries of vegetation
    phenology statistics.
    
    last modified March 2020
    
    Parameters
    ----------
    - da:  xarray.Dataset
    - doy: xarray.DataArray
        Day-of-year values for each time step in the 'time'
        dim on 'da'. e.g doy=da.time.dt.dayofyear
    - stats: list
        list of phenological statistics to return. Regardless of
        the metrics returned, all statistics are
        calculated due to inter-dependicises between metrics.
        Options include:
            SOS = DOY of start of season
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
        xarray.Dataset 
        
    """
    da = ds.to_array().squeeze()
    
    # Capture input band names in order to drop these if drop=True
    if drop:
        bands_to_drop=list(ds.data_vars)
        print(f'Dropping bands {bands_to_drop}')
    
    vpos = _vpos(da)   
    ipos = _ipos(da, vpos)
    pos = _pos(doy, ipos)
    trough = _trough(da)
    aos =_aos(vpos, trough)

    ratio = _ratio(da, trough, aos)
    sos = _sos(ratio, doy, ipos)
    eos = _eos(ratio, doy, sos)
    isos = _isos(doy, sos)
    ieos = _ieos(doy, eos)
    los = _los(eos, sos, da)
    rog = _rog(doy, sos, eos, da, vpos, isos, pos, sos)
    ros = _ros(da, ieos, vpos, eos, pos)
    sw = _sw(da, doy, sos, eos)
    vsos = _vsos(da, isos)
    veos = _veos(da, ieos)

    stats = list(sos,pos[0],eos, vsos, vpos, veos, los, ampl, ios, rog, ros, sw)
    
    # Dictionary containing the statistics
    stat_dict = {'SOS':sos,
                 'EOS':eos,
                 'vSOS':vsos,
                 'vPOS':vpos,
                 'vEOS':veos,
                 'LOS':los,
                 'AOS':ampl,
                 'IOS':ios,
                 'ROG':rog,
                 'ROS':ros,
                 'SW':sw}
    
    # Add as a new variable in dataset
    for stat in stats:
        output_band_name = stat
        
        #only keep the stats asked for
        stat_keep = stat_dict.get(str(stat)) 
        ds[output_band_name] = index_array

    # Once all indexes are calculated, drop input bands if drop=True
    if drop: 
        ds = ds.drop(bands_to_drop)

    # Return input dataset with added water index variable
    return ds

