# training_data_parallel.py

'''
Description: This file contains a set of python functions for extracting
training data from the ODC in parallel across many cpus. This can be useful
when a very large number of training data polygons or points nned to be
queried to create a training data sample.  

License: The code in this notebook is licensed under the Apache License, 
Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Digital Earth 
Africa data is licensed under the Creative Commons by Attribution 4.0 
license (https://creativecommons.org/licenses/by/4.0/).

Contact: If you need assistance, please post a question on the Open Data 
Cube Slack channel (http://slack.opendatacube.org/) or on the GIS Stack 
Exchange (https://gis.stackexchange.com/questions/ask?tags=open-data-cube) 
using the `open-data-cube` tag (you can view previously asked questions 
here: https://gis.stackexchange.com/questions/tagged/open-data-cube).

If you would like to report an issue with this script, you can file one on 
Github https://github.com/digitalearthafrica/deafrica-sandbox-notebooks/issues

Last modified: April 2020

'''

import numpy as np
import xarray as xr
import geopandas as gpd
from copy import deepcopy
import datacube
from datacube.utils import geometry
import multiprocessing as mp
from tqdm import tqdm
from datacube_stats.statistics import GeoMedian
import sys
import os

sys.path.append('../Scripts')
from deafrica_datahandling import mostcommon_crs, load_ard
from deafrica_bandindices import calculate_indices
from deafrica_spatialtools import xr_rasterize
from deafrica_classificationtools import sklearn_flatten, HiddenPrints

def get_training_data_for_shp(gdf,
                              index,
                              row,
                              out_arrs,
                              out_vars,
                              products,
                              dc_query,
                              custom_func=None,
                              field=None,
                              calc_indices=None,
                              reduce_func=None,
                              drop=True,
                              zonal_stats=None):
    
    """
    Function to extract data from the ODC for training a machine learning classifier using a geopandas 
    geodataframe of labelled geometries.  The function will loop through each row in a geopandas
    dataframe and extract ODC data for the region encompassed by the geometry. 
    This function provides a number of pre-defined methods for producing training data, 
    including calcuating band indices, reducing time series using several summary statistics, 
    and/or generating zonal statistics across polygons.  The 'custom_func' parameter provides 
    a method for the user to supply a function for generating features rather than using the
    pre-defined methods.
     
    Parameters
    ----------
    gdf : geopandas geodataframe
        geometry data in the form of a geopandas geodataframe
    products : list
        a list of products to load from the datacube. 
        e.g. ['ls8_usgs_sr_scene', 'ls7_usgs_sr_scene']
    dc_query : dictionary
        Datacube query object, should not contain lat and long (x or y)
        variables as these are supplied by the 'gdf' variable
    field : string 
        A string containing the name of column with class labels. 
        Field must contain numeric values.
    out_arrs : multiprocessing.Manager.list() 
        An empty Manage.list into which the training data arrays are stored.
        This is handled by the 'get_training_data_parallel' function.
    out_vars : multiprocessing.Manager.list() 
        An empty list into which the data varaible names are stored.
        This is handled by the 'get_training_data_parallel' function.
    custom_func : function, optional 
        A custom function for generating feature layers. If this parameter
        is set, all other options (excluding 'zonal_stats'), will be ignored.
        The result of the 'custom_func' must be a single xarray dataset 
        containing 2D coordinates (i.e x, y - no time dimension). The custom function
        has access to the datacube dataset extracted using the 'dc_query' params,
        along with access to the 'dc_query' dictionary itself, which could be used
        to load other products besides those specified under 'products'.
    calc_indices: list, optional
        If not using a custom func, the this parameter provides a method for
        calculating any number of remote sensing indices (e.g. `['NDWI', 'NDVI']`).
    reduce_func : string, optional 
        Function to reduce the data from multiple time steps to
        a single timestep. Options are 'mean', 'median', 'std',
        'max', 'min', 'geomedian'.  Ignored if custom_func is provided.
    drop : boolean, optional , 
        If this variable is set to True, and 'calc_indices' are supplied, the
        spectral bands will be dropped from the dataset leaving only the
        band indices as data variables in the dataset. Default is True.
    zonal_stats : string, optional
        An optional string giving the names of zonal statistics to calculate 
        for each polygon. Default is None (all pixel values are returned). Supported 
        values are 'mean', 'median', 'max', 'min', and 'std'. Will work in 
        conjuction with a 'custom_func'.


    Returns
    --------
    Two lists, a list of numpy.arrays containing classes and extracted data for 
    each pixel or polygon, and another containing the data variable names.

    """
    
    #prevent function altering dictionary kwargs
    dc_query = deepcopy(dc_query)
    
    # remove dask chunks if supplied as using mulitprocessing
    # for parallization  
    if 'dask_chunks' in dc_query.keys():
        dc_query.pop('dask_chunks', None)
    
    #connec to to datacube
    dc = datacube.Datacube(app='training_data')

    # set up query based on polygon (convert to WGS84)
    geom = geometry.Geometry(
        gdf.geometry.values[index].__geo_interface__, geometry.CRS(
            'epsg:4326'))

    #print(geom)    
    q = {"geopolygon": geom}

    # merge polygon query with user supplied query params
    dc_query.update(q)
    
    # Identify the most common projection system in the input query
    output_crs = mostcommon_crs(dc=dc, product=products, query=dc_query)

    #load_ard doesn't handle geomedians
    if 'ga_ls8c_gm_2_annual' in products:
        ds = dc.load(product='ga_ls8c_gm_2_annual', **dc_query)
        ds = ds.where(ds!=0, np.nan)

    else:
        # load data
        with HiddenPrints():
            ds = load_ard(dc=dc,
                          products=products,
                          output_crs=output_crs,
                          **dc_query)

    # create polygon mask
    with HiddenPrints():
        mask = xr_rasterize(gdf.iloc[[index]], ds)

    #mask dataset
    ds = ds.where(mask)

    # Use custom function for training data if it exists
    if custom_func is not None:
        with HiddenPrints():
            data = custom_func(ds)

    else:       
        #first check enough variables are set to run functions
        if (len(ds.time.values) > 1) and (reduce_func==None):
                raise ValueError("You're dataset has "+ str(len(ds.time.values)) + 
                                 " time-steps, please provide a reduction function," +
                                 " e.g. reduce_func='mean'")

        if calc_indices is not None:
            #determine which collection is being loaded
            if 'level2' in products[0]:
                collection = 'c2'
            elif 'gm' in products[0]:
                collection = 'c2'
            elif 'sr' in products[0]:
                collection = 'c1'
            elif 's2' in products:
                collection = 's2'

            if len(ds.time.values) > 1:

                if reduce_func in ['mean','median','std','max','min']:
                    with HiddenPrints():
                        data = calculate_indices(ds,
                                                 index=calc_indices,
                                                 drop=drop,
                                                 collection=collection)
                        method_to_call = getattr(data, reduce_func)
                        data = method_to_call(dim='time')

                elif reduce_func == 'geomedian':
                    data = GeoMedian().compute(ds)
                    with HiddenPrints():
                        data = calculate_indices(data,
                                                 index=calc_indices,
                                                 drop=drop,
                                                 collection=collection)

                else:
                    raise Exception(reduce_func+ " is not one of the supported" + 
                        " reduce functions ('mean','median','std','max','min', 'geomedian')")

            else:
                with HiddenPrints():
                    data = calculate_indices(ds,
                                             index=calc_indices,
                                             drop=drop,
                                             collection=collection)

        # when band indices are not required, reduce the
        # dataset to a 2d array through means or (geo)medians
        if calc_indices is None:

            if len(ds.time.values) > 1:

                if reduce_func == 'geomedian':
                    data = GeoMedian().compute(ds)

                elif reduce_func in ['mean','median','std','max','min']:
                    method_to_call = getattr(ds, reduce_func)
                    data = method_to_call('time')
            else:
                data = ds.squeeze()

    # compute in case we have dask arrays
    if 'dask_chunks' in dc_query.keys():
        data = data.compute()
    
    if zonal_stats is None:
        # If no zonal stats were requested then extract all pixel values
        flat_train = sklearn_flatten(data)
        # Make a labelled array of identical size
        flat_val = np.repeat(row[field], flat_train.shape[0])
        stacked = np.hstack((np.expand_dims(flat_val, axis=1), flat_train))
        
    elif zonal_stats in ['mean','median','std','max','min']:
        method_to_call = getattr(data, zonal_stats)
        flat_train = method_to_call()
        flat_train = flat_train.to_array()
        stacked = np.hstack((row[field], flat_train))

    else:
        raise Exception(zonal_stats+ " is not one of the supported" +
                        " reduce functions ('mean','median','std','max','min')")

    # Append training data and labels to list
    out_arrs.append(stacked)
    out_vars.append([field] + list(data.data_vars))


def get_training_data_parallel(ncpus, gdf, products, dc_query,
         custom_func=None, field=None, calc_indices=None,
         reduce_func=None, drop=True, zonal_stats=None):
    
        """
        Function passing the 'get_training_data_f0r_shp' function
        to a mulitprocessing.Pool.
        Inherits variables from 'main()'.
        
        """
        # instantiate lists that can be shared across processes
        manager = mp.Manager()
        results = manager.list()
        column_names = manager.list()

        #progress bar
        pbar = tqdm(total=len(gdf))
        def update(*a):
            pbar.update()

        with mp.Pool(ncpus) as pool:
            for index, row in gdf.iterrows():
                pool.apply_async(get_training_data_for_shp,
                                           [gdf,
                                            index,
                                            row,
                                            results,
                                            column_names,
                                            products,
                                            dc_query,
                                            custom_func,
                                            field,
                                            calc_indices,
                                            reduce_func,
                                            drop,
                                            zonal_stats], callback=update)

            pool.close()
            pool.join()
            pbar.close()

        return column_names, results

def main(ncpus, gdf, products, dc_query,
         custom_func=None, field=None, calc_indices=None,
         reduce_func=None, drop=True, zonal_stats=None):
    
    """
    This function executes the training data functions and tidies the results
    into a 'model_input' object containing stacked training data arrays
    with all NaNs removed. In the instance where ncpus=1, a serial version of the
    function will be run instead of passing the functions to mp.Pool.
    
    Parameters
    ----------
    ncpus : int
        The number of cpus/processes over which to parallelize the gathering
        of training data (only if ncpus is > 1). Use 'mp.cpu_count()' to determine the number of
        cpus available on a machine.
    
    See function 'get_training_data_for_shp' for descriptions of other input
    parameters.
    
    Returns
    --------
    Two lists, one contains a list of numpy.arrays with classes and extracted data for 
    each pixel or polygon, and another containing the data variable names.

    
    """
    #set up some print statements
    if custom_func is not None:
            print("Reducing data using user supplied custom function")   
    if calc_indices is not None and custom_func is None:
            print("Calculating indices: " + str(calc_indices))
    if reduce_func is not None and custom_func is None:
            print("Reducing data using: " + reduce_func)
    if zonal_stats is not None:
            print("Taking zonal statistic: "+ zonal_stats)
    
    if ncpus == 1:
        #progress indicator
        i = 0
        # list to store results
        results = []
        column_names = []
        # loop through polys and extract training data
        for index, row in gdf.iterrows():
            print(" Feature {:04}/{:04}\r".format(i + 1, len(gdf)), 
                  end='')

            get_training_data_for_shp(gdf,
                                    index,
                                    row,
                                    results,
                                    column_names,
                                    products,
                                    dc_query,
                                    custom_func,
                                    field,
                                    calc_indices,
                                    reduce_func,
                                    drop,
                                    zonal_stats)
            i+=1
            

    else:
        column_names, results = get_training_data_parallel(ncpus=ncpus,
                                        gdf=gdf,
                                        products=products,
                                        dc_query=dc_query,
                                        custom_func=custom_func,
                                        field=field,
                                        calc_indices=calc_indices,
                                        reduce_func=reduce_func,
                                        drop=drop,
                                        zonal_stats=zonal_stats)
    
    column_names=column_names[0]

    #Stack the extracted training data for each feature into a single array
    model_input = np.vstack(results)
    print(f'\nOutput training data has shape {model_input.shape}')

    # Remove any potential nans
    model_input = model_input[~np.isnan(model_input).any(axis=1)]
    print("Removed NaNs, cleaned input shape: ", model_input.shape)
    
    return column_names, model_input
