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
import dask

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
    
    #prevent function altering dictionary kwargs
    dc_query = deepcopy(dc_query)
    
    #connecto to datacube
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


def get_training_data_parallel(gdf, products, dc_query,
         custom_func=None, field=None, calc_indices=None,
         reduce_func=None, drop=True, zonal_stats=None):

        # instantiate lists that can be shared across processes
        manager = mp.Manager()
        results = manager.list()
        column_names = manager.list()
        
        print('extracting training data...')
        #progress bar
        pbar = tqdm(total=len(gdf))
        def update(*a):
            pbar.update()
        
        with mp.Pool(mp.cpu_count()-1) as pool:
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

def main(gdf, products, dc_query,
         custom_func=None, field=None, calc_indices=None,
         reduce_func=None, drop=True, zonal_stats=None):
    
    column_names, results = get_training_data_parallel(gdf=gdf,
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


#---------DASK VERSION THAT WORKS-----
#This works but only returns the arrays, not the columns.
# delayed_results = [dask.delayed(parallel_train.get_training_data_for_shp)(input_data[0:4],
#                                 index,
#                                 row,
#                                 products,
#                                 query,
#                                 custom_func,
#                                 field,
#                                 calc_indices,
#                                 reduce_func,
#                                 drop,
#                                 zonal_stats) for index, row in input_data[0:4].iterrows()]

# results = compute(*delayed_results)