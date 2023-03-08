# this for jsd
from PIL import Image
from scipy.spatial.distance import jensenshannon
import numpy as np
import os
from rpy2.situation import get_r_home
os.environ["R_HOME"] = get_r_home()
import rpy2.robjects as robjects


MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT = 116.08, 39.66, 116.69, 40.27 # Beijing center
GRID_RESOLUTION = 500
TARGET_EPSG = 32650

def raster_usage_count(MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT, GRID_RESOLUTION, TARGET_EPSG):

    input_file_path = "C:/Users/Bened/Documents/Git/Master-Thesis/Geolife/user_135.geojson"
    output_file_path = "user_135.tif"
    robjects.r("""
        library(sf)

        raster_usage_count_R <- function(MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT, 
                        resolution, TARGET_EPSG,
                        INPUT_FILE_PATH, OUTPUT_FILE_PATH){
                
                # transform bounding box from 4326 to 3035
                pts <- matrix(c(MIN_LNG, MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG, 
                                MAX_LAT, MAX_LNG, MIN_LAT, MIN_LNG, MIN_LAT), ncol=2, byrow=TRUE)
                polygon_ext <- st_polygon(list(pts)) |> st_sfc(crs=4326) |> st_transform(TARGET_EPSG)
                extent <- st_as_sf(polygon_ext)|> terra::ext()
                
                
                raster_template <- terra::rast(crs= terra::crs("epsg:32650"), 
                                                res=resolution,
                                                extent=extent,
                                                vals=0) 
                
                input_ls <- read_sf(INPUT_FILE_PATH)|> st_transform(TARGET_EPSG)
                
                count_raster <- raster_template 

                for (i in input_ls$tid){
                  line <- input_ls|> dplyr::filter(tid == i) |> dplyr::select(geometry)
                # if line only consists of at least two points
                if (nrow(sf::st_cast(line, "POINT")) > 1) {
                        count_raster_temp <- terra::rasterize(line, raster_template, background=0)
                        count_raster <- count_raster + count_raster_temp
                  }
                }
                
                terra::writeRaster(count_raster, OUTPUT_FILE_PATH, overwrite=TRUE)
            }
        """)

    raster_usage_count_R = robjects.globalenv["raster_usage_count_R"]
    raster_usage_count_R(MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT, \
                            GRID_RESOLUTION, TARGET_EPSG, input_file_path, output_file_path)
    
raster_usage_count(MIN_LNG, MIN_LAT, MAX_LNG, MAX_LAT, GRID_RESOLUTION, TARGET_EPSG)
    


def similarity_of_tifs():
    alt_path = "input1.tif"
    base_path = "input2.tif"

    # Open image
    im = Image.open(alt_path)
    alt_counts = np.array(im)
    # this is for cells whithout any value (no traj passed through there) -> set to 0
    alt_counts[alt_counts<0] = 0
    alt_counts = alt_counts

    im = Image.open(base_path)
    base_counts = np.array(im)
    # this is for cells whithout any value (no traj passed through there) -> set to 0
    base_counts[base_counts<0] = 0
    base_counts = base_counts

    jsd = jensenshannon(alt_counts.flatten(), base_counts.flatten())