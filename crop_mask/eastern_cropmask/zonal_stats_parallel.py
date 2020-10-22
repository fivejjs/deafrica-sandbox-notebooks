import fiona
from rasterstats import zonal_stats
import multiprocessing as mp
from shapely.geometry import mapping, shape

def zs_parallel(shp,raster,statistics,out_shp,ncpus):

    """
    Summarizing raster datasets based on vector geometries in
    parallel
    
    Parameters
    ----------
    shp : str
        Path to shapefile on disk that contain the polygons over
        which the zonal statistics will be calculated
    raster: str
        Path to the raster from which the statistics are calculated
    out_shp: str
        Path to export shapefile containing zonal statistcs.
    ncpus: int
        number of cores to parallelize the operations over
    
    Returns
    -------
    gdf : Geopandas GeoDataFrame
    
    """
    
    #yields n sized chunks from list l (used for splitting task to multiple processes)
    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    #calculates zonal stats and adds results to a dictionary
    def worker(z,raster,d):	
        z_stats = zonal_stats(z,raster, stats=statistics)	
        for i in range(0,len(z_stats)):
            d[z[i]['id']]=z_stats[i]

    #write output polygon
    def write_output(zones, out_shp,d):
        #copy schema and crs from input and add new fields for each statistic			
        schema = zones.schema.copy()
        crs = zones.crs
        for stat in statistics:			
            schema['properties'][stat] = 'float'

        with fiona.open(out_shp, 'w', 'ESRI Shapefile', schema, crs) as output:
            for elem in zones:
                for stat in statistics:			
                    elem['properties'][stat]=d[elem['id']][stat]
                output.write({'properties':elem['properties'],'geometry': mapping(shape(elem['geometry']))})
    
    with fiona.open(shp) as zones:
        jobs = []

        #create manager dictionary (polygon ids=keys, stats=entries) where multiple processes can write without conflicts
        man = mp.Manager()	
        d = man.dict()	

        #split zone polygons into 10 chunks for parallel processing and call worker() for each. 
        # Adjust 10 to be number of cores you want to use for optimal performance.
        split = chunks(zones, len(zones)//ncpus)
        for z in split:
            p = mp.Process(target=worker,args=(z, raster,d))
            p.start()
            jobs.append(p)

        #wait that all chunks are finished
        [j.join() for j in jobs]

        write_output(zones,out_shp,d)		
