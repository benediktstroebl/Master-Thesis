
import geopandas as gp
import pandas as pd
from shapely.geometry import Point

# read PERSON_IDs from test set
test_ids = []
with open("test_set_user_ids.txt", "r") as f:
    for line in f:
        test_ids.append(int(line.strip()))

# Load data from secure Veracrypt partition
raw_full_trip_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/raw_full_trip.geojson")
raw_points_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/od_points.geojson")
tesselation_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/tessellation.geojson")

# remove ids from test set
raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(test_ids) == False]
raw_points_gdf = raw_points_gdf[raw_points_gdf['uid'].isin(test_ids) == False]

# Convert datetime columns to strftime format for plotting with Geopandas
raw_full_trip_gdf['TRIP_START'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
raw_full_trip_gdf['TRIP_END'] = pd.to_datetime(raw_full_trip_gdf.TRIP_END, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
raw_points_gdf['datetime'] = pd.to_datetime(raw_points_gdf.datetime, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')

# Extract weekday and date
raw_full_trip_gdf['TRIP_WD'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%A')
raw_full_trip_gdf['TRIP_DATE'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')

# Filter for users with more than N trips
n = 1
less_than_n_trips = raw_full_trip_gdf.groupby('PERSON_ID').count().sort_values('TRIP_ID').reset_index()
less_than_n_trips = less_than_n_trips[less_than_n_trips['TRIP_ID'] >= n]
raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(less_than_n_trips.PERSON_ID)]

# Convert SP and EP to POINT object 
raw_full_trip_gdf['TRIP_SP'] = raw_full_trip_gdf['TRIP_SP'].apply(lambda x: Point(tuple(map(float, x.replace("(", "").replace(")", "").split(",")))))
raw_full_trip_gdf['TRIP_EP'] = raw_full_trip_gdf['TRIP_EP'].apply(lambda x: Point(tuple(map(float, x.replace("(", "").replace(")", "").split(",")))))

# Create two separate gdfs for start and endpoints
raw_trip_sp_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_EP'], axis=1)
raw_trip_sp_gdf['LONG'], raw_trip_sp_gdf['LAT']  = raw_trip_sp_gdf['TRIP_SP'].apply(lambda x: x.x), raw_trip_sp_gdf['TRIP_SP'].apply(lambda x: x.y)
raw_trip_ep_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_SP'], axis=1)
raw_trip_ep_gdf['LONG'], raw_trip_ep_gdf['LAT']  = raw_trip_ep_gdf['TRIP_EP'].apply(lambda x: x.x), raw_trip_ep_gdf['TRIP_EP'].apply(lambda x: x.y)

# Convert EP and SP file to GeoDataFrame
raw_trip_ep_gdf = gp.GeoDataFrame(raw_trip_ep_gdf, geometry="TRIP_EP", crs = 4326)
raw_trip_sp_gdf = gp.GeoDataFrame(raw_trip_sp_gdf, geometry="TRIP_SP", crs = 4326)

# Project all GDFs to CRS 3035
raw_trip_ep_gdf = raw_trip_ep_gdf.to_crs(3035)
raw_trip_sp_gdf = raw_trip_sp_gdf.to_crs(3035)
raw_full_trip_gdf = raw_full_trip_gdf.to_crs(3035)
raw_points_gdf = raw_points_gdf.to_crs(3035)
tesselation_gdf = tesselation_gdf.to_crs(3035)

# drop SP and EP columns from trip file
raw_full_trip_gdf = raw_full_trip_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1)