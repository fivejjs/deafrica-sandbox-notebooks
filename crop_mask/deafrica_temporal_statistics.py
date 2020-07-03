# deafrica_temporal_statistics.py
"""
This script contains functions for calculating per-pixel
temporal summary statistics on a timeseries stored in a
xarray.DataArray.

There are two primary functions in this script:
    1. xr_phenology: calculates land-surface phenology
                    metrics on a time series of a vegetation index

    2. temporal_statistics: calculates various generic summary
                            statistics on any timeseries.

Other function support these two primary functions.

Functions:
---------
    allNaN_arg
    fast_completion
    smooth
    various phenology statistics
    xr_phenology
    temporal_statistics
    
----------  
TO DO:
- Handle dask arrays once xarray releases version 0.16
- Implement intergal-of-season statistic for xr_phenology

"""

from deafrica_datahandling import first, last
import sys
import dask
import numpy as np
import xarray as xr
import hdstats

sys.path.append("../Scripts")


def allNaN_arg(da, dim, stat):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    float and then masks the offending cells.

    Params
    ------
    xarr : xarray.DataArray
    dim : str, 
            Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str,
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    Returns
    ------
    xarray.DataArray
    """
    # generate a mask where entire axis along dimension is NaN
    mask = da.isnull().all(dim)

    if stat == "max":
        y = da.fillna(float(da.min() - 1))
        y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == "min":
        y = da.fillna(float(da.max() + 1))
        y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y


def fast_completion(arr):
    """
    gap-fill a timeseries
    """
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    i, j = np.meshgrid(np.arange(idx.shape[0]),
                       np.arange(idx.shape[1]),
                       indexing="ij")
    dat = arr[i[:, :, np.newaxis], j[:, :, np.newaxis], idx]
    if np.isnan(np.sum(dat[:, :, 0])):
        fill = np.nanmean(dat, axis=-1)
        for t in range(dat.shape[-1]):
            mask = np.isnan(dat[:, :, t])
            if mask.any():
                dat[mask, t] = fill[mask]
            else:
                break
    return dat


def smooth(arr, k=3):
    """
    Apply the scipy.signal.weiner filter
    to a time-series
    """
    return wiener(arr, (1, 1, k))


def _vpos(da):
    """
    vPOS = Value at peak of season
    """
    return da.max("time")


def _pos(da):
    """
    POS = DOY of peak of season
    """
    return da.isel(time=da.argmax("time")).time.dt.dayofyear


def _trough(da):
    """
    Trough = Minimum value
    """
    return da.min("time")


def _aos(vpos, trough):
    """
    AOS = Amplitude of season
    """
    return vpos - trough


def _vsos(da, pos, method_sos="first"):
    """
    vSOS = Value at the start of season
    Params
    -----
    da : xarray.DataArray
    method_sos : str, 
        If 'first' then vSOS is estimated
        as the first positive slope on the
        greening side of the curve. If 'median',
        then vSOS is estimated as the median value
        of the postive slopes on the greening side
        of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    greenup = da.where(da.time < pos.time)
    # find the first order slopes
    green_deriv = greenup.differentiate("time")
    # find where the first order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv > 0)
    # positive slopes on greening side
    pos_greenup = greenup.where(pos_green_deriv)
    # find the median
    median = pos_greenup.median("time")
    # distance of values from median
    distance = pos_greenup - median

    if method_sos == "first":
        # find index (argmin) where distance is most negative
        idx = allNaN_arg(distance, "time", "min").astype("int16")

    if method_sos == "median":
        # find index (argmin) where distance is smallest absolute value
        idx = allNaN_arg(xr.ufuncs.fabs(distance), "time",
                         "min").astype("int16")

    return pos_greenup.isel(time=idx)


def _sos(vsos):
    """
    SOS = DOY for start of season
    """
    return vsos.time.dt.dayofyear


def _veos(da, pos, method_eos="last"):
    """
    vEOS = Value at the start of season
    Params
    -----
    method_eos : str
        If 'last' then vEOS is estimated
        as the last negative slope on the
        senescing side of the curve. If 'median',
        then vEOS is estimated as the 'median' value
        of the negative slopes on the senescing 
        side of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    senesce = da.where(da.time > pos.time)
    # find the first order slopes
    senesce_deriv = senesce.differentiate("time")
    # find where the fst order slope is postive
    neg_senesce_deriv = senesce_deriv.where(senesce_deriv < 0)
    # negative slopes on senescing side
    neg_senesce = senesce.where(neg_senesce_deriv)
    # find medians
    median = neg_senesce.median("time")
    # distance to the median
    distance = neg_senesce - median

    if method_eos == "last":
        # index where last negative slope occurs
        idx = allNaN_arg(distance, "time", "min").astype("int16")

    if method_eos == "median":
        # index where median occurs
        idx = allNaN_arg(xr.ufuncs.fabs(distance), "time",
                         "min").astype("int16")

    return neg_senesce.isel(time=idx)


def _eos(veos):
    """
    EOS = DOY for end of seasonn
    """
    return veos.time.dt.dayofyear


def _los(da, eos, sos):
    """
    LOS = Length of season (in DOY)
    """
    los = eos - sos
    # handle negative values
    los = xr.where(
        los >= 0,
        los,
        da.time.dt.dayofyear.values[-1] +
        (eos.where(los < 0) - sos.where(los < 0)),
    )

    return los


def _rog(vpos, vsos, pos, sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos - vsos) / (pos - sos)


def _ros(veos, vpos, eos, pos):
    """
    ROG = Rate of Senescing (Days)
    """
    return (veos - vpos) / (eos - pos)


def xr_phenology(
    da,
    stats=[
        "SOS",
        "POS",
        "EOS",
        "Trough",
        "vSOS",
        "vPOS",
        "vEOS",
        "LOS",
        "AOS",
        "ROG",
        "ROS",
    ],
    method_sos="first",
    method_eos="last",
    smooth=None,
):
    """
    Obtain land surface phenology metrics from an
    xarray.DataArray containing a timeseries of a 
    vegetation index like NDVI.

    last modified June 2020

    Parameters
    ----------
    da :  xarray.DataArray
        DataArray should contain a 2D or 3D time series of a
        vegetation index like NDVI, EVI
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
    method_sos : str 
        If 'first' then vSOS is estimated as the first positive 
        slope on the greening side of the curve. If 'median',
        then vSOS is estimated as the median value of the postive
        slopes on the greening side of the curve.
    method_eos : str
        If 'last' then vEOS is estimated as the last negative slope
        on the senescing side of the curve. If 'median', then vEOS is
        estimated as the 'median' value of the negative slopes on the
        senescing side of the curve.
    complete : bool
        If True, the timeseries will be completed (gap filled) using
        hdstats.fast_completion()
    smooth : str
        If 'weiner', the timeseries will be smoothed using the
        scipy.signal.weiner filter with a window size of 3.  If 'rolling_mean', 
        then timeseries is smoothed using a rolling mean with a window size of 3


    Outputs
    -------
        xarray.Dataset containing variables for the selected 
        phenology statistics 

    """
    # Check inputs before running calculations
    if dask.is_dask_collection(da):
        raise TypeError(
            "Dask arrays are not currently supported by this function, " +
            "run da.compute() before passing dataArray.")

    if method_sos not in ("median", "first"):
        raise ValueError("method_sos should be either 'median' or 'first'")

    if method_eos not in ("median", "last"):
        raise ValueError("method_eos should be either 'median' or 'last'")

    # If stats supplied is not a list, convert to list.
    stats = stats if isinstance(stats, list) else [stats]

    # complete the timeseries (remove NaNs)
    # grab coords etc
    x, y, time, attrs = da.x, da.y, da.time, da.attrs

    # reshape to satisfy function
    da = da.transpose("y", "x", "time").values

    # complete timeseries
    print("Completing...")
    da = fast_completion(da)

    if smooth is not None:

        if smooth == "rolling_mean":
            print("   Smoothing with rolling mean...")
            da = xr.DataArray(
                da,
                attrs=attrs,
                coords={
                    "x": x,
                    "y": y,
                    "time": time
                },
                dims=["y", "x", "time"],
            )

            da = da.rolling(time=3, min_periods=1).mean()

        if smooth == "weiner":
            print("   Smoothing with weiner filter...")
            da = smooth(da)
            # place back into xarray
            da = xr.DataArray(
                da,
                attrs=attrs,
                coords={
                    "x": x,
                    "y": y,
                    "time": time
                },
                dims=["y", "x", "time"],
            )
    else:
        da = xr.DataArray(
            da,
            attrs=attrs,
            coords={
                "x": x,
                "y": y,
                "time": time
            },
            dims=["y", "x", "time"],
        )

    # remove any remaining all-NaN pixels
    mask = da.isnull().all("time")
    da = da.where(~mask, other=0)

    # calculate the statistics
    print("      Phenology...")
    vpos = _vpos(da)
    pos = _pos(da)
    trough = _trough(da)
    aos = _aos(vpos, trough)
    vsos = _vsos(da, pos, method_sos=method_sos)
    sos = _sos(vsos)
    veos = _veos(da, pos, method_eos=method_eos)
    eos = _eos(veos)
    los = _los(da, eos, sos)
    rog = _rog(vpos, vsos, pos, sos)
    ros = _ros(veos, vpos, eos, pos)

    # Dictionary containing the statistics
    stats_dict = {
        "SOS": sos.astype(np.int16),
        "EOS": eos.astype(np.int16),
        "vSOS": vsos.astype(np.float32),
        "vPOS": vpos.astype(np.float32),
        "Trough": trough.astype(np.float32),
        "POS": pos.astype(np.int16),
        "vEOS": veos.astype(np.float32),
        "LOS": los.astype(np.int16),
        "AOS": aos.astype(np.float32),
        "ROG": rog.astype(np.float32),
        "ROS": ros.astype(np.float32),
    }

    # intialise dataset with first statistic
    ds = stats_dict[stats[0]].to_dataset(name=stats[0])

    # add the other stats to the dataset
    for stat in stats[1:]:
        print("         " + stat)
        stats_keep = stats_dict.get(stat)
        ds[stat] = stats_dict[stat]

    return ds


def temporal_statistics(da, stats):
    """
    Obtain generic temporal statistics using the hdstats temporal library:
    https://github.com/daleroberts/hdstats/blob/master/hdstats/ts.pyx

    last modified June 2020

    Parameters
    ----------
    da :  xarray.DataArray
        DataArray should contain a 3D time series.
    stats : list
        list of temporal statistics to calculate.
        Options include:
            'discordance' = 
            'f_std' = std of discrete fourier transform coefficients, returns
                      three layers: f_std_n1, f_std_n2, f_std_n3
            'f_mean' = mean of discrete fourier transform coefficients, returns
                       three layers: f_mean_n1, f_mean_n2, f_mean_n3
            'f_median' = median of discrete fourier transform coefficients, returns
                         three layers: f_median_n1, f_median_n2, f_median_n3
            'mean_change' = mean of discrete difference along time dimension
            'median_change' = median of discrete difference along time dimension
            'abs_change' = mean of absolute discrete difference along time dimension
            'complexity' = 
            'central_diff' = 
            'num_peaks' : The number of peaks in the timeseries, defined with a local
                          window of size 10.  NOTE: This statistic is very slow to calculate.

    Outputs
    -------
        xarray.Dataset containing variables for the selected 
        temporal statistics 

    """
    # If stats supplied is not a list, convert to list.
    stats = stats if isinstance(stats, list) else [stats]
    
    # grab all the attributes of the xarray
    x, y, time, attrs = da.x, da.y, da.time, da.attrs

    # deal with any all-NaN pixels by filling with 0's
    mask = da.isnull().all("time")
    da = da.where(~mask, other=0)

    # reshape to satisfy functions
    da = da.transpose("y", "x", "time").values

    # complete timeseries
    print("Completing...")
    da = fast_completion(da)

    stats_dict = {
        "discordance": lambda da: hdstats.discordance(da, n=10),
        "f_std": lambda da: hdstats.fourier_std(da, n=3, step=5),
        "f_mean": lambda da: hdstats.fourier_mean(da, n=3, step=5),
        "f_median": lambda da: hdstats.fourier_median(da, n=3, step=5),
        "mean_change": lambda da: hdstats.mean_change(da),
        "median_change": lambda da: hdstats.median_change(da),
        "abs_change": lambda da: hdstats.mean_abs_change(da),
        "complexity": lambda da: hdstats.complexity(da),
        "central_diff": lambda da: hdstats.mean_central_diff(da),
        "num_peaks": lambda da: hdstats.number_peaks(da, 10),
    }


    print("   Statistics:")
    # if one of the fourier functions is first (or only)
    # stat in the list then we need to deal with this
    if stats[0] in ("f_std", "f_median", "f_mean"):
        print("      " + stats[0])
        stat_func = stats_dict.get(str(stats[0]))
        zz = stat_func(da)
        n1 = zz[:, :, 0]
        n2 = zz[:, :, 1]
        n3 = zz[:, :, 2]

        # intialise dataset with first statistic
        ds = xr.DataArray(n1,
                          attrs=attrs,
                          coords={
                              "x": x,
                              "y": y
                          },
                          dims=["y", "x"]).to_dataset(name=stats[0] + "_n1")

        # add other datasets
        for i, j in zip([n2, n3], ["n2", "n3"]):
            ds[stats[0] + "_" + j] = xr.DataArray(i,
                                                  attrs=attrs,
                                                  coords={
                                                      "x": x,
                                                      "y": y
                                                  },
                                                  dims=["y", "x"])
    else:
        # simpler if first function isn't fourier transform
        first_func = stats_dict.get(str(stats[0]))
        print("      " + stats[0])
        ds = first_func(da)

        # convert back to xarray dataset
        ds = xr.DataArray(ds,
                          attrs=attrs,
                          coords={
                              "x": x,
                              "y": y
                          },
                          dims=["y", "x"]).to_dataset(name=stats[0])

    # loop through the other functions
    for stat in stats[1:]:
        print("      " + stat)

        # handle the fourier transform examples
        if stat in ("f_std", "f_median", "f_mean"):
            stat_func = stats_dict.get(str(stat))
            zz = stat_func(da)
            n1 = zz[:, :, 0]
            n2 = zz[:, :, 1]
            n3 = zz[:, :, 2]

            for i, j in zip([n1, n2, n3], ["n1", "n2", "n3"]):
                ds[stat + "_" + j] = xr.DataArray(i,
                                                  attrs=attrs,
                                                  coords={
                                                      "x": x,
                                                      "y": y
                                                  },
                                                  dims=["y", "x"])

        else:
            # Select a stats function from the dictionary
            # and add to the dataset
            stat_func = stats_dict.get(str(stat))
            ds[stat] = xr.DataArray(stat_func(da),
                                    attrs=attrs,
                                    coords={
                                        "x": x,
                                        "y": y
                                    },
                                    dims=["y", "x"])

    return ds
