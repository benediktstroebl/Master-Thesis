import pandas as pd
import geopandas as gp
from shapely import Point
import numpy as np

def select_n_random_users_from_dataframes(n, raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf):
    """Select n random users from the dataset.

    Args:
        n (_type_): Number of users to select.
    """
    # Select n random users
    users = np.random.choice(raw_full_trip_gdf['PERSON_ID'].unique(), n, replace=False)
    # Filter dataframes
    raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(users)]
    raw_trip_sp_gdf = raw_trip_sp_gdf[raw_trip_sp_gdf['PERSON_ID'].isin(users)]
    raw_trip_ep_gdf = raw_trip_ep_gdf[raw_trip_ep_gdf['PERSON_ID'].isin(users)]
    return raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf


def load_geolife(n_rand_users=None, n_trajs=None, only_toy_data=False):
    # Load geolife data with EPSG:32650 (China)
    print("Reading geolife geojson file...")
    geolife_raw_gdf = gp.read_file("../data/geolife/geolife_splitted.geojson", geometry='geometry', rows=n_trajs)
    print("Done.")

    # For debug and testing, we can use these user_ids
    if only_toy_data:
        toy_person_ids = [72, 107,  11,  26, 169]
        geolife_raw_gdf = geolife_raw_gdf.query('user_id in @toy_person_ids')

    # Load geolife tesselated data with EPSG:32650 (China)
    geolife_tesselation_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/tessellation_geolife.geojson", geometry='geometry').to_crs('EPSG:32650')

    # Replace traj_id with increasing integer values
    geolife_raw_gdf['traj_id'] = range(0, len(geolife_raw_gdf))

    # Create SP and EP columns
    geolife_raw_gdf['TRIP_SP'] = geolife_raw_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    geolife_raw_gdf['TRIP_EP'] = geolife_raw_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    # Rename columns and drop unnecessary columns
    geolife_raw_gdf.rename(columns={
        'start_t': 'TRIP_START', 
        'end_t': 'TRIP_END', 
        'traj_id': 'TRIP_ID',
        'length': 'TRIP_LEN_IN_MTRS',
        'user_id': 'PERSON_ID'}, inplace=True)
        
    geolife_raw_gdf.drop(columns=['direction'], axis=1, inplace=True)

    # Extract weekday and date
    geolife_raw_gdf['TRIP_WD'] = pd.to_datetime(geolife_raw_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%A')
    geolife_raw_gdf['TRIP_DATE'] = pd.to_datetime(geolife_raw_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')

    # Create three separate dataframes for each SP, EP, and full trip
    geolife_raw_sp_gdf = geolife_raw_gdf.drop(['geometry', 'TRIP_EP'], axis=1).copy().set_geometry('TRIP_SP')
    geolife_raw_ep_gdf = geolife_raw_gdf.drop(['geometry', 'TRIP_SP'], axis=1).copy().set_geometry('TRIP_EP')
    geolife_raw_full_trip_gdf = geolife_raw_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1).copy().set_geometry('geometry')

    assert len(geolife_raw_gdf) == len(geolife_raw_sp_gdf) == len(geolife_raw_ep_gdf) == len(geolife_raw_full_trip_gdf)

    if n_rand_users is not None:
        geolife_raw_full_trip_gdf, geolife_raw_sp_gdf, geolife_raw_ep_gdf = select_n_random_users_from_dataframes(n_rand_users, geolife_raw_full_trip_gdf, geolife_raw_sp_gdf, geolife_raw_ep_gdf)

    # Print number of trajectories in dataset
    print("Number of trajectories in loaded dataset: {}".format(len(geolife_raw_full_trip_gdf)))

    # Number of users in dataset
    print("Number of users in loaded dataset: {}".format(len(geolife_raw_full_trip_gdf['PERSON_ID'].unique())))

    return geolife_raw_full_trip_gdf, geolife_raw_sp_gdf, geolife_raw_ep_gdf, geolife_tesselation_gdf


def load_freemove(n_rand_users=None, n_trajs=None, hide_test_users=True):
    # read PERSON_IDs from test set
    test_ids = []
    with open("test_set_user_ids.txt", "r") as f:
        for line in f:
            test_ids.append(int(line.strip()))

    # Load data from secure Veracrypt partition
    raw_full_trip_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/raw_full_trip.geojson")
    raw_points_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/od_points.geojson")
    tesselation_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/tessellation.geojson")

    if hide_test_users:
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
    raw_trip_ep_gdf = gp.GeoDataFrame(raw_trip_ep_gdf, geometry="TRIP_EP", crs="EPSG:4326")
    raw_trip_sp_gdf = gp.GeoDataFrame(raw_trip_sp_gdf, geometry="TRIP_SP", crs="EPSG:4326")

    # Project all GDFs to CRS 3035
    raw_trip_ep_gdf = raw_trip_ep_gdf.to_crs("EPSG:3035")
    raw_trip_sp_gdf = raw_trip_sp_gdf.to_crs("EPSG:3035")
    raw_full_trip_gdf = raw_full_trip_gdf.to_crs("EPSG:3035")
    raw_points_gdf = raw_points_gdf.to_crs("EPSG:3035")
    tesselation_gdf = tesselation_gdf.to_crs("EPSG:3035")

    # drop SP and EP columns from trip file
    raw_full_trip_gdf = raw_full_trip_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1)

    return raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf