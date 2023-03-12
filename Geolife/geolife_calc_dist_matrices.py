import sys
sys.path.append('..')

import attack
import data_loader as dl
from numpy import save


# Load splitted data
print("Reading splitted geolife geojson file...")
raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf = dl.load_geolife()
assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf)
print("Done.")


# calc dist matrix for splitted data
print("Calculating distance matrix for splitted data...")
A = attack.cdist(raw_full_trip_gdf.geometry)
save('A_splitted.npy', A)
del A
print("Done.")


# Load smooth_generalized data
print("Reading smooth_generalized geolife geojson file...")
raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf = dl.load_geolife(data_type='smoothed_generalized')
assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf)
print("Done.")

# calc dist matrix
print("Calculating distance matrix for smooth_generalized data...")
A = attack.cdist(raw_full_trip_gdf.geometry)
save('A_smooth_generalized.npy', A)
del A
print("Done.")


# Load smoothed data
print("Reading smoothed geolife geojson file...")
raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf = dl.load_geolife(data_type='smoothed')
assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf)
print("Done.")


# calc dist matrix
print("Calculating distance matrix for smoothed data...")
A = attack.cdist(raw_full_trip_gdf.geometry)
save('A_smooth.npy', A)
del A
print("Done.")

# Load generalized data
print("Reading generalized geolife geojson file...")
raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf = dl.load_geolife(data_type='generalized')
assert len(raw_full_trip_gdf) == len(raw_trip_sp_gdf) == len(raw_trip_ep_gdf)
print("Done.")

# calc dist matrix
print("Calculating distance matrix for generalized data...")
A = attack.cdist(raw_full_trip_gdf.geometry)
save('A_generalized.npy', A)
del A
print("Done.")

