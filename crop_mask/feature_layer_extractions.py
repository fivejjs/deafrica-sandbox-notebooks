
import richdem as rd
import pyproj
from datacube.utils.geometry import assign_crs
from odc.algo import xr_reproject
import datacube
import numpy as np
import sys
import xarray as xr

sys.path.append('../Scripts')
from deafrica_bandindices import calculate_indices
from deafrica_temporal_statistics import xr_phenology, temporal_statistics
from deafrica_classificationtools import HiddenPrints
from deafrica_datahandling import load_ard


def xr_terrain(da, attribute=None):
    """
    Using the richdem package, calculates terrain attributes
    on a DEM stored in memory as an xarray.DataArray 
    
    Params
    -------
    da : xr.DataArray
    attribute : str
        One of the terrain attributes that richdem.TerrainAttribute()
        has implemented. e.g. 'slope_riserun', 'slope_percentage', 'aspect'.
        See all option here:  
        https://richdem.readthedocs.io/en/latest/python_api.html#richdem.TerrainAttribute
        
    """
    #remove time if its there
    da = da.squeeze()
    #convert to richdem array
    rda = rd.rdarray(da.data, no_data=da.attrs['nodata'])
    #add projection and geotransform
    rda.projection=pyproj.crs.CRS(da.attrs['crs']).to_wkt()
    rda.geotransform = da.geobox.affine.to_gdal()
    #calulate attribute
    attrs = rd.TerrainAttribute(rda, attrib=attribute)

    #return as xarray DataArray
    return xr.DataArray(attrs,
                        attrs=da.attrs,
                        coords={'x':da.x, 'y':da.y},
                        dims=['y', 'x'])


def phenology_features(ds):
    dc = datacube.Datacube(app='training')
    data = calculate_indices(ds,
                             index=['NDVI'],
                             drop=True,
                             collection='s2')
    
    #ndvi = data.NDVI.mean(['x','y'])
    
    #temporal stats
    ts = temporal_statistics(data.NDVI,
                       stats=['f_mean', 'abs_change','discordance'
                              'complexity','central_diff'])

    #ts = xr_phenology(ndvi, complete='linear')
    
    #rainfall climatology
    print('rainfall...')
    chirps = assign_crs(xr.open_rasterio('data/CHIRPS/CHPclim_sum.nc'),  crs='epsg:4326')
    chirps = xr_reproject(chirps,ds.geobox,"mode")
    chirps = chirps.to_dataset(name='chirps')
    #chirps = chirps.mean(['x','y'])
    
    #slope
    print('slope...')
    slope = dc.load(product='srtm', like=ds.geobox).squeeze()
    slope = slope.elevation
    slope = xr_terrain(slope, 'slope_riserun')
    slope = slope.to_dataset(name='slope')
    #slope = slope.mean(['x','y'])
    
    #Surface reflectance results
    print("SR..")
    sr = ds.median('time')
    #sr = ds.mean(['x','y']).median('time')
    print('Merging...')
    result = xr.merge([ts, sr, chirps,slope], compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)
    
    return result.squeeze()

def two_epochs(ds):
    dc = datacube.Datacube(app='training')
    
    print('epoch 1')
    epoch1 = calculate_indices(ds,
                             index=['NDVI'],
                             drop=False,
                             collection='s2')
    
    epoch1 = epoch1.median('time')

    q = {
    'geopolygon':ds.geobox.extent,
    'time': ('2019-06', '2019-12'),
    'measurements': [
                     'blue',
                     'green',
                     'red',
                     'nir_1',
                    ],
    'resolution': (-20, 20),
    'group_by' :'solar_day',
    'output_crs':'epsg:6933'}
    

    print('epoch 2')    
    ds2 = load_ard(dc=dc,products=['s2_l2a'],**q)    
    
    epoch2 = calculate_indices(ds2,
                             index=['NDVI'],
                             drop=False,
                             collection='s2')
    
    epoch2 = epoch2.median('time')
    
    epoch2 = epoch2.rename({
                     'blue':'blue_2',
                     'green':'green_2',
                     'red':'red_2',
                     'nir_1':'nir_1_2',
                     'NDVI':'NDVI_2'
                      })

    print('slope...')
    slope = dc.load(product='srtm', like=ds.geobox).squeeze()
    slope = slope.elevation
    slope = xr_terrain(slope, 'slope_riserun')
    slope = slope.to_dataset(name='slope')
    
    print('Merging...')
    result = xr.merge([epoch1,epoch2,slope], compat='override')
    result = assign_crs(result, crs=ds.geobox.crs)
    print(result)
    return result.squeeze()