#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import geopandas as gp
from pathlib import Path
from shapely import Point


# In[81]:


# Load trajectories from all user from individual csv files
path = '..\data\geolife\split_trajectories' # Path to split trajectories
all_files = Path(path).glob('*.csv')

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

geolife_raw_gdf = pd.concat(li, axis=0, ignore_index=True)


# In[82]:


# Load geolife data with EPSG:32650 (China)
geolife_raw_gdf = gp.GeoDataFrame(geolife_raw_gdf, geometry=gp.GeoSeries.from_wkt(geolife_raw_gdf['geometry']), crs='EPSG:32650')


# In[ ]:

# Load geolife tesselated data with EPSG:32650 (China)
geolife_tesselation_gdf = gp.read_file("W:/Master-Thesis-Repository/data/freemove_dlr_data/tessellation_geolife.geojson", crs='EPSG:32650', geometry='geometry')


# In[83]:


# Replace traj_id with increasing integer values
geolife_raw_gdf['traj_id'] = range(0, len(geolife_raw_gdf))


# In[84]:


# Create SP and EP columns
geolife_raw_gdf['TRIP_SP'] = geolife_raw_gdf.geometry.apply(lambda x: Point(x.coords[0]))
geolife_raw_gdf['TRIP_EP'] = geolife_raw_gdf.geometry.apply(lambda x: Point(x.coords[-1]))


# In[86]:


# Rename columns and drop unnecessary columns
geolife_raw_gdf.rename(columns={
    'start_t': 'TRIP_START', 
    'end_t': 'TRIP_END', 
    'traj_id': 'TRIP_ID',
    'length': 'TRIP_LEN_IN_MTRS',
    'user_id': 'PERSON_ID'}, inplace=True)
    
geolife_raw_gdf.drop(columns=['direction'], axis=1, inplace=True)


# In[87]:


# Extract weekday and date
geolife_raw_gdf['TRIP_WD'] = pd.to_datetime(geolife_raw_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%A')
geolife_raw_gdf['TRIP_DATE'] = pd.to_datetime(geolife_raw_gdf.TRIP_START, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d')


# In[92]:


# Create three separate dataframes for each SP, EP, and full trip
geolife_raw_sp_gdf = geolife_raw_gdf.drop(['geometry', 'TRIP_EP'], axis=1).copy().set_geometry('TRIP_SP')
geolife_raw_ep_gdf = geolife_raw_gdf.drop(['geometry', 'TRIP_SP'], axis=1).copy().set_geometry('TRIP_EP')
geolife_raw_full_trip_gdf = geolife_raw_gdf.drop(['TRIP_SP', 'TRIP_EP'], axis=1).copy().set_geometry('geometry')

assert len(geolife_raw_gdf) == len(geolife_raw_sp_gdf) == len(geolife_raw_ep_gdf) == len(geolife_raw_full_trip_gdf)

