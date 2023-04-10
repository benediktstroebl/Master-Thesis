import pandas as pd
import geopandas as gp
from shapely import Point, LineString
import numpy as np
from datetime import timedelta

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


def load_geolife(n_rand_users=None, n_trajs=None, only_toy_data=False, data_type='raw', min_n_trips_per_user=1, tessellation_diameter=200, rand_n_week_period=None, min_trip_length=None, min_nr_points=1, upper_quantile_nr_points=1.0, only_2008=True):
    # Load geolife data with EPSG:32650 (China)
    if data_type == 'raw':
        print("Reading raw geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_raw.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'private':
        print("Reading private geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_raw_private.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'splitted':
        print("Reading splitted geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_splitted.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'smoothed':
        print("Reading smoothed geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_splitted_smooth.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'generalized':
        print("Reading generalized geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_splitted_generalized.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'smoothed_generalized':
        print("Reading smoothed and generalized geolife geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/geolife/geolife_splitted_smooth_generalized.geojson", geometry='geometry', rows=n_trajs)
    print("Done.")

    # Replace traj_id with increasing integer values
    raw_full_trip_gdf['traj_id'] = range(0, len(raw_full_trip_gdf))

    # Extract 2008 data
    if only_2008:
        raw_full_trip_gdf = raw_full_trip_gdf[pd.to_datetime(raw_full_trip_gdf.start_t, format='%Y-%m-%d %H:%M:%S').dt.year == 2008].copy()

    # For debug and testing, we can use these user_ids
    if only_toy_data:
        toy_person_ids = [72, 107,  11,  26, 169]
        raw_full_trip_gdf = raw_full_trip_gdf.query('user_id in @toy_person_ids')

    # Load geolife tesselated data with EPSG:32650 (China)
    geolife_tesselation_gdf = gp.read_file("../data/geolife/tessellation_geolife_" + str(tessellation_diameter) + ".geojson", geometry='geometry').to_crs('EPSG:32650')

    # Filter for min nr of points in each linestring
    raw_full_trip_gdf.geometry.fillna(value=LineString(), inplace=True)
    raw_full_trip_gdf['NR_POINTS'] = raw_full_trip_gdf.geometry.apply(lambda x: len(x.coords))
    raw_full_trip_gdf = raw_full_trip_gdf.query('NR_POINTS > @min_nr_points').copy()
    upper_quantile = raw_full_trip_gdf.NR_POINTS.quantile(upper_quantile_nr_points)
    raw_full_trip_gdf = raw_full_trip_gdf.query('NR_POINTS <= @upper_quantile').copy()

    # Create SP and EP columns
    raw_full_trip_gdf['TRIP_SP'] = raw_full_trip_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    raw_full_trip_gdf['TRIP_EP'] = raw_full_trip_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    # Rename columns and drop unnecessary columns
    raw_full_trip_gdf.rename(columns={
        'start_t': 'TRIP_START', 
        'end_t': 'TRIP_END', 
        'traj_id': 'TRIP_ID',
        'length': 'TRIP_LEN_IN_MTRS',
        'user_id': 'PERSON_ID'}, inplace=True)
        
    raw_full_trip_gdf.drop(columns=['direction', 'NR_POINTS'], axis=1, inplace=True)

    # Filter for users with more than N trips
    less_than_n_trips = raw_full_trip_gdf.groupby('PERSON_ID').count().sort_values('TRIP_ID').reset_index()
    less_than_n_trips = less_than_n_trips[less_than_n_trips['TRIP_ID'] >= min_n_trips_per_user]
    raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(less_than_n_trips.PERSON_ID)]

    # Extract weekday and date
    raw_full_trip_gdf['TRIP_WD'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%A')
    raw_full_trip_gdf['TRIP_DATE'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')

    # Extract random n-week period from the dataset raw_full_trip_gdf
    if rand_n_week_period is not None:
        # Get random 4-week period
        start_date = pd.to_datetime(raw_full_trip_gdf.TRIP_DATE.unique()).min()
        end_date = pd.to_datetime(raw_full_trip_gdf.TRIP_DATE.unique()).max()
        days = rand_n_week_period * 7
        rand_start_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days - days))
        rand_end_date = rand_start_date + timedelta(days=days)
        # Filter raw_full_trip_gdf
        raw_full_trip_gdf = raw_full_trip_gdf[(raw_full_trip_gdf['TRIP_DATE'] >= rand_start_date.strftime('%Y-%m-%d')) & (raw_full_trip_gdf['TRIP_DATE'] <= rand_end_date.strftime('%Y-%m-%d'))]
    
    # Filter for min trip length if provided
    if min_trip_length is not None:
        raw_full_trip_gdf = raw_full_trip_gdf.query('TRIP_LEN_IN_MTRS > @min_trip_length').copy()


    # Calculate trip duration in minutes
    raw_full_trip_gdf['TRIP_DURATION_IN_MINS'] = (pd.to_datetime(raw_full_trip_gdf.TRIP_END, format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S')).dt.total_seconds() / 60

    # Create three separate dataframes for each SP, EP, and full trip
    raw_trip_sp_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_EP'], axis=1).copy().set_geometry('TRIP_SP')
    raw_trip_ep_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_SP'], axis=1).copy().set_geometry('TRIP_EP')
    geolife_raw_full_trip_gdf = raw_full_trip_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1).copy().set_geometry('geometry')

    assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf) == len(geolife_raw_full_trip_gdf)

    if n_rand_users is not None:
        geolife_raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf = select_n_random_users_from_dataframes(n_rand_users, geolife_raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf)

    # Print number of trajectories in dataset
    print("Number of trajectories in loaded dataset: {}".format(len(geolife_raw_full_trip_gdf)))

    # Number of users in dataset
    print("Number of users in loaded dataset: {}".format(len(geolife_raw_full_trip_gdf['PERSON_ID'].unique())))

    return geolife_raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, geolife_tesselation_gdf


def load_freemove(n_rand_users=None, n_trajs=None, hide_test_users=True, data_type='raw', min_n_trips_per_user=1, tessellation_diameter=200, rand_n_week_period=None, min_trip_length=None, min_nr_points=1, upper_quantile_nr_points=1.0):
    # read PERSON_IDs from test set
    test_ids = []
    with open("../freemove/test_set_user_ids.txt", "r") as f:
        for line in f:
            test_ids.append(int(line.strip()))

    # Load data from secure Veracrypt partition
    if data_type == 'raw':
        print("Reading raw freemove geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/freemove/freemove_raw.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'private':
        print("Reading private freemove geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/freemove/freemove_raw_private.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'smoothed':
        print("Reading smoothed freemove geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/freemove/freemove_smooth.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'generalized':
        print("Reading generalized freemove geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/freemove/freemove_generalized.geojson", geometry='geometry', rows=n_trajs)
    elif data_type == 'smoothed_generalized':
        print("Reading smoothed and generalized freemove geojson file...")
        raw_full_trip_gdf = gp.read_file("../data/freemove/freemove_smooth_generalized.geojson", geometry='geometry', rows=n_trajs)
    print("Done.")  

    # Load tesselation data
    tesselation_gdf = gp.read_file("../data/freemove/tessellation_freemove_" + str(tessellation_diameter) + ".geojson").to_crs(epsg=3035)

    # Replace traj_id with increasing integer values
    raw_full_trip_gdf['traj_id'] = range(0, len(raw_full_trip_gdf))

    # Filter for min nr of points in each linestring
    raw_full_trip_gdf.geometry.fillna(value=LineString(), inplace=True)
    raw_full_trip_gdf['NR_POINTS'] = raw_full_trip_gdf.geometry.apply(lambda x: len(x.coords))
    raw_full_trip_gdf = raw_full_trip_gdf.query('NR_POINTS > @min_nr_points').copy()
    upper_quantile = raw_full_trip_gdf.NR_POINTS.quantile(upper_quantile_nr_points)
    raw_full_trip_gdf = raw_full_trip_gdf.query('NR_POINTS <= @upper_quantile').copy()

    # Create SP and EP columns
    raw_full_trip_gdf['TRIP_SP'] = raw_full_trip_gdf.geometry.apply(lambda x: Point(x.coords[0]))
    raw_full_trip_gdf['TRIP_EP'] = raw_full_trip_gdf.geometry.apply(lambda x: Point(x.coords[-1]))

    # Rename columns and drop unnecessary columns
    raw_full_trip_gdf.rename(columns={
        'start_t': 'TRIP_START', 
        'end_t': 'TRIP_END', 
        'traj_id': 'TRIP_ID',
        'length': 'TRIP_LEN_IN_MTRS',
        'user_id': 'PERSON_ID'}, inplace=True)
        
    raw_full_trip_gdf.drop(columns=['direction', 'NR_POINTS'], axis=1, inplace=True)

    if hide_test_users:
        # remove ids from test set
        raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(test_ids) == False]

    # Filter for users with more than N trips
    less_than_n_trips = raw_full_trip_gdf.groupby('PERSON_ID').count().sort_values('TRIP_ID').reset_index()
    less_than_n_trips = less_than_n_trips[less_than_n_trips['TRIP_ID'] >= min_n_trips_per_user]
    raw_full_trip_gdf = raw_full_trip_gdf[raw_full_trip_gdf['PERSON_ID'].isin(less_than_n_trips.PERSON_ID)]

    # Extract weekday and date
    raw_full_trip_gdf['TRIP_WD'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%A')
    raw_full_trip_gdf['TRIP_DATE'] = pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')

    # Extract random n-week period from the dataset raw_full_trip_gdf
    if rand_n_week_period is not None:
        # Get random 4-week period
        start_date = pd.to_datetime(raw_full_trip_gdf.TRIP_DATE.unique()).min()
        end_date = pd.to_datetime(raw_full_trip_gdf.TRIP_DATE.unique()).max()
        days = rand_n_week_period * 7
        rand_start_date = start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days - days))
        rand_end_date = rand_start_date + timedelta(days=days)
        # Filter raw_full_trip_gdf
        raw_full_trip_gdf = raw_full_trip_gdf[(raw_full_trip_gdf['TRIP_DATE'] >= rand_start_date.strftime('%Y-%m-%d')) & (raw_full_trip_gdf['TRIP_DATE'] <= rand_end_date.strftime('%Y-%m-%d'))]
        
    # Filter for min trip length if provided
    if min_trip_length is not None:
        raw_full_trip_gdf = raw_full_trip_gdf.query('TRIP_LEN_IN_MTRS > @min_trip_length').copy()

    # Calculate trip duration in minutes
    raw_full_trip_gdf['TRIP_DURATION_IN_MINS'] = (pd.to_datetime(raw_full_trip_gdf.TRIP_END, format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(raw_full_trip_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S')).dt.total_seconds() / 60

    # Create three separate dataframes for each SP, EP, and full trip
    raw_trip_sp_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_EP'], axis=1).copy().set_geometry('TRIP_SP')
    raw_trip_ep_gdf = raw_full_trip_gdf.drop(['geometry', 'TRIP_SP'], axis=1).copy().set_geometry('TRIP_EP')
    raw_full_trip_gdf = raw_full_trip_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1).copy().set_geometry('geometry')

    assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf)

    if n_rand_users is not None:
        raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf = select_n_random_users_from_dataframes(n_rand_users, raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf)

    # Print number of trajectories in dataset
    print("Number of trajectories in loaded dataset: {}".format(len(raw_full_trip_gdf)))

    # Number of users in dataset
    print("Number of users in loaded dataset: {}".format(len(raw_full_trip_gdf['PERSON_ID'].unique())))

    return raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf