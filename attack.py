import geopandas as gp
import pandas as pd
from tqdm import tqdm
import numpy as np
import libpysal
import itertools
from joblib import Parallel, delayed
import random
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import tslearn.metrics


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

def getGroundTruth(full_trip_gdf):
    # Get ground truth labels
    df = full_trip_gdf.copy()
    df['ID'] = df.sort_values('TRIP_ID').groupby('PERSON_ID').ngroup() # Sort TRIP ID ascending and set cluster id corresponding to PERSON_ID
    ground_truth = df.sort_values('TRIP_ID').ID.to_list()
    return ground_truth

def evaluate(clustering, full_trip_gdf):
    # Get ground truth labels
    ground_truth = getGroundTruth(full_trip_gdf)

    # # Not symmetric and not accounting for chance
    # print(f"Accuracy@1: {getAccuracyAtOne(ground_truth, clustering):.3f}")
    # print(f"Precision: {getPrecision(ground_truth, clustering):.3f}")
    # print(f"Recall: {getRecall(ground_truth, clustering):.3f}")
    # print(f"F1: {getF1Score(ground_truth, clustering):.3f}")
    print(f"Homogeneity: {metrics.homogeneity_score(ground_truth, clustering):.3f}")
    print(f"Completeness: {metrics.completeness_score(ground_truth, clustering):.3f}")

    # All of these metrics are symmetric and some of them are accounting for chance depending on the number of classes and clusters present in the data
    print(f"V-measure: {metrics.v_measure_score(ground_truth, clustering):.3f}")
    print(f"Rand index: {metrics.rand_score(ground_truth, clustering):.3f}")
    print(f"ARI: {metrics.adjusted_rand_score(ground_truth, clustering):.3f}")
    print(f"MI: {metrics.mutual_info_score(ground_truth, clustering):.3f}")
    print(f"NMI: {metrics.normalized_mutual_info_score(ground_truth, clustering):.3f}")
    print(f"AMI: {metrics.adjusted_mutual_info_score(ground_truth, clustering):.3f}")

def LCSS(traj1_linestr, traj2_linestr, eps=10):
    """This function takes in two GeoSeries and takes the top entry linestring. It then calculates the Least Common Sub-Sequence metric for these two and returns the value.

    Args:
        traj1_linestr (_type_): _description_
        traj2_linestr (_type_): _description_
        eps (int, optional): This can be interpreted as the distance in meters between two points compared of the subsequences. Defaults to 10.

    Returns:
        _type_: float
    """
    assert isinstance(traj1_linestr, gp.GeoSeries), f"traj1_linestr is of type {type(traj1_linestr)}, need to be GeoSeries"
    assert isinstance(traj2_linestr, gp.GeoSeries), f"traj2_linestr is of type {type(traj2_linestr)}, need to be GeoSeries"
    assert len(traj1_linestr) > 0, "traj1_linestr is empty"
    assert len(traj2_linestr) > 0, "traj2_linestr is empty"
    
    s1 = traj1_linestr.iloc[0].coords
    s2 = traj2_linestr.iloc[0].coords

    return tslearn.metrics.lcss(s1, s2, eps=eps)

def match_boundary_points_with_tessellation(raw_trip_sp_gdf, raw_trip_ep_gdf, tesselation_gdf):
    """This function matches the boundary points of the raw trips with the tesselation. 

    Args:
        raw_trip_sp_gdf (_type_): _description_
        raw_trip_ep_gdf (_type_): _description_
        tesselation_gdf (_type_): _description_

    Returns:
        _type_: This function returns two data frames, one for the start points and one for the end points. These data frames contain the tile_id of the tesselation that the point is located in.
    """
    # SP
    # Spatial join points to polygons
    gdf_sp = gp.sjoin(
        tesselation_gdf[["tile_id", "geometry"]],
        raw_trip_sp_gdf,
        how="inner"
    ).drop('index_right', axis=1)

    # Spatial join points to polygons
    gdf_ep = gp.sjoin(
        tesselation_gdf[["tile_id", "geometry"]],
        raw_trip_ep_gdf,
        how="inner"
    ).drop('index_right', axis=1)

    return gdf_sp, gdf_ep


def extract_trips_that_start_end_in_tessellation(raw_full_trip_gdf, raw_trip_sp_gdf, raw_trip_ep_gdf, gdf_sp, gdf_ep):
    gdf_sp_ids = gdf_sp.TRIP_ID
    gdf_ep_ids = gdf_ep.TRIP_ID

    full_trip_gdf = raw_full_trip_gdf.query("TRIP_ID in @gdf_sp_ids and TRIP_ID in @gdf_ep_ids")
    trip_sp_gdf = raw_trip_sp_gdf.query("TRIP_ID in @gdf_ep_ids and TRIP_ID in @gdf_sp_ids")
    trip_ep_gdf = raw_trip_ep_gdf.query("TRIP_ID in @gdf_sp_ids and TRIP_ID in @gdf_ep_ids")

    gdf_sp = gdf_sp.query("TRIP_ID in @gdf_ep_ids and TRIP_ID in @gdf_sp_ids")
    gdf_ep = gdf_ep.query("TRIP_ID in @gdf_sp_ids and TRIP_ID in @gdf_ep_ids")

    assert len(full_trip_gdf) == len(trip_sp_gdf) == len(trip_ep_gdf) == len(gdf_sp) == len(gdf_ep) == len(set(trip_sp_gdf.TRIP_ID).intersection(set(trip_ep_gdf.TRIP_ID))), "Not equal length." # this last intersection checks that for all unique trip ids we have exactly ONE SP and EP

    print(f"Number of trips that start and end wihin tessellation area: {len(full_trip_gdf)}")
    print(f"Number of trips outside and therefore dropped: {len(raw_full_trip_gdf) - len(full_trip_gdf)}")

    return full_trip_gdf, trip_sp_gdf, trip_ep_gdf, gdf_sp, gdf_ep

def build_trip_chain_mapping(gdf_sp, gdf_ep, INFLOW_HR_DIFF_THRESHOLD=4, HR_DIFF_THRESHOLD=8):
    """This function returns a list of trip chains that are continued trips that happened subsequent to and from same tile within a given time threshold.

    Args:
        gdf_sp (_type_): _description_
        gdf_ep (_type_): _description_
        inflow_hr_diff_threshold (int, optional): _description_. Defaults to 4.
        hr_diff_threshold (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """

    # Calculate mapping of continued trips that happened subsequent to and from same tile
    mapping_cont_trips = []
    for index, trip in tqdm(gdf_ep.sort_values('TRIP_ID').iterrows(), total=len(gdf_ep)):
        te_1_id = trip.TRIP_ID
        te_1_tid = trip.tile_id
        te_1_dt = pd.to_datetime(trip['TRIP_END'], format='%Y-%m-%d %H:%M:%S')
        ts_1_dt = pd.to_datetime(trip['TRIP_START'], format='%Y-%m-%d %H:%M:%S')

        inflow = gdf_ep.query("tile_id == @te_1_tid").copy()
        inflow['TRIP_END'] = inflow.TRIP_END.astype('datetime64[ns]')
        inflow['TRIP_START'] = inflow.TRIP_START.astype('datetime64[ns]')
        inflow['INFLOW_HR_DIFF'] = inflow.TRIP_END.apply(lambda x: (x - te_1_dt).total_seconds()/3600)
        inflow = inflow.query("(INFLOW_HR_DIFF <= @INFLOW_HR_DIFF_THRESHOLD) and (INFLOW_HR_DIFF >= @INFLOW_HR_DIFF_THRESHOLD)") # Take trips 
        inflow = inflow.query("(TRIP_START > @te_1_dt) or (@ts_1_dt > TRIP_END)") # Ignore trips that have happened simultaneously

        # if more than one trip has arrived in +- hour window, then do not merge this trip
        if len(inflow) > 1:
            continue

        
        # Get all trips that started from same tile as t_1 has ended in
        ts_2 = gdf_sp.query("tile_id == @te_1_tid").copy()

        # get difference between two trips hours (seconds divided by 3600 gets hours)
        ts_2['TRIP_START'] = ts_2.TRIP_START.astype('datetime64[ns]')
        ts_2['TRIP_END'] = ts_2.TRIP_END.astype('datetime64[ns]')
    
        ts_2['hr_diff'] = ts_2['TRIP_START'].apply(lambda x: (x - te_1_dt).total_seconds()/3600)

        # Only consider trips that started within a certain time after the initial trip ended in the same tessellation tile
        ts_2 = ts_2[(ts_2['hr_diff'].astype(str).astype(float) <= HR_DIFF_THRESHOLD) & (ts_2['hr_diff'].astype(str).astype(float) >= 0)]

        # Only consider trips that are not simultaneously
        ts_2 = ts_2.query("(TRIP_START > @te_1_dt) or (@ts_1_dt > TRIP_END)")

        # Only consider connection if exactly one trip started from same tile in time window
        if len(ts_2) == 1:
            mapping_cont_trips.append({
                'TRIP_ID': te_1_id,
                'TRIP_ID_CONT': ts_2.TRIP_ID.iloc[0]
            })

    return mapping_cont_trips


def evaluate_trip_chaining(mapping_cont_trips, full_trip_gdf):
    """This function evaluates the trip chaining by checking if the chained trips are from the same person.

    Args:
        mapping_cont_trips (_type_): Dictionary of trip ids that are chained. Output of build_trip_chain_mapping()
        full_trip_gdf (_type_): The full trip gdf that contains all trips.
    Returns:
        _type_: None
    """
    mistakes = []
    for conn in mapping_cont_trips:
        trip_ids = [conn['TRIP_ID'],  conn['TRIP_ID_CONT']]
        unique_person = full_trip_gdf.query("TRIP_ID in @trip_ids").PERSON_ID.nunique()

        if unique_person > 1:
            mistakes.append(full_trip_gdf.query("TRIP_ID in @trip_ids"))


    print(f"Number of edges (matched) between trips: {len(mapping_cont_trips)}")
    print(f"Number of wrong matches: {len(mistakes)}")


def getTripChain(trip_id, mapping_cont_trips, chain=[]):
    """ Recursive function that returns a list for all chained trips for a give orig trip_id


    Args:
        trip_id (_type_): _description_
        chain (list, optional): _description_. Defaults to [].
        mapping_cont_trips (_type_): Mapping of continued trips. Output of build_trip_chain_mapping().

    Returns:
        _type_: _description_
    """
    if type(trip_id) == str:
        trip_id = int(trip_id)

    # add orig trip_id to output list
    if len(chain) == 0:
        chain.append(trip_id)

    # recursively find all chained trips originating from the orig trip_id
    for edge in mapping_cont_trips:
        if edge['TRIP_ID'] == trip_id:
            chain.append(edge['TRIP_ID_CONT'])
            getTripChain(edge['TRIP_ID_CONT'], mapping_cont_trips, chain)
            
        
    return chain

def merge_trips_from_matching(gdf_sp, mapping_cont_trips, full_trip_gdf):
    """This function merges trips that are chained together from the matching done in build_trip_chain_mapping().

    Args:
        gdf_sp (_type_): GeoDataFrame of start points.
        mapping_cont_trips (_type_): Mapping of continued trips. Output of build_trip_chain_mapping().
        full_trip_gdf (_type_): GeoDataFrame of all trips.

    Returns:
        _type_: GeoDataFrame of merged trips.
    """
    # Get trip chain for each trip (Start Point)
    print("Building trip chains...")
    trip_chains = [getTripChain(trip, mapping_cont_trips, chain=[]) for trip in tqdm(gdf_sp.TRIP_ID)]
    print("Done.")

    # Sort for longest chain first
    trip_chains.sort(key = len, reverse = True)

    # Create dictionary to store mappings for evaluation
    trip_concat_dict = {}

    covered_trips = []
    merged_trips_gdf = []
    print("Merging trips...")
    for chain in tqdm(trip_chains, total=len(trip_chains)):
        # Check if any of the trips in the current chain has already been merged as part of another chain
        # Since we start with the longest chain and iterate through descending sorted list, we only retain the complete chains
        if set(chain).intersection(set(covered_trips)):
            continue

        # add trip chain to dict for evaluation later
        trip_concat_dict[chain[0]] = chain[1:]
        
        # add all trip ids part of current chain to list so that every trip is only contained in longest chain of it
        covered_trips += chain

        trips = full_trip_gdf.query("TRIP_ID in @chain").sort_values("TRIP_START")
        trips["temp"] = 1

        trips = trips.groupby('temp').agg(list).reset_index(drop=True).rename(columns={'TRIP_ID': 'TRIP_ID_CHAIN'})

        trips["wkt_trip"] = trips['geometry'].apply(lambda x: ", ".join([str(i) for i in x]).replace("), LINESTRING (", ", "))
        trips['TRIP_START'] = trips['TRIP_START'].apply(lambda x: min(x))
        trips['TRIP_END'] = trips['TRIP_END'].apply(lambda x: max(x))
        trips['TRIP_LEN_IN_MTRS'] = trips['TRIP_LEN_IN_MTRS'].apply(lambda x: sum(x))
        #trips['TRIP_DURATION_IN_SECS'] = trips['TRIP_DURATION_IN_SECS'].apply(lambda x: sum(x))
        trips['TRIP_WD'] = trips['TRIP_WD'].apply(lambda x: x[0]) # see below
        trips['TRIP_DATE'] = trips['TRIP_DATE'].apply(lambda x: x[0]) # see below
        trips['TRIP_ID'] = trips['TRIP_ID_CHAIN'].apply(lambda x: x[0]) # assign trip_id of first trip in chain to concatenated trip
        # This is the TRIP_ID of the last trip in the chain to be concatenated
        trips['TRIP_ID_LAST'] = trips['TRIP_ID_CHAIN'].apply(lambda x: x[-1]) 

        # Note: Here we are assigning the PERSON_ID of the first trip to the concatenated trip. This of course can be erroneous if the concatenation itself is wrong
        trips['PERSON_ID'] = trips['PERSON_ID'].apply(lambda x: x[0])
        trips = trips.drop(['geometry', 'TRIP_ID_CHAIN'], axis=1)

        trips = gp.GeoDataFrame(trips, geometry=gp.GeoSeries.from_wkt(trips['wkt_trip'])).drop('wkt_trip', axis=1)

        merged_trips_gdf.append(trips)
    print("Done.")

    trip_merged_gdf = pd.concat(merged_trips_gdf)

    print(f"Number of trips that were matched at least once: {len(set(covered_trips))}/{len(set(gdf_sp.TRIP_ID))}")

    # Concatenate all trips that were unmerged with the merged trips into a new gdf
    print("Concatenating MERGED and UNMERGED trips...")
    unmerged_trips = full_trip_gdf.query("TRIP_ID not in @covered_trips")
    full_trips_concat_gdf = pd.concat([unmerged_trips, trip_merged_gdf])
    full_trips_concat_gdf['TRIP_ID_FIRST'] = full_trips_concat_gdf['TRIP_ID'] # This is the same as TRIP_ID
    print("Done.")

    # Assign TRIP_ID as TRIP_ID_LAST in case TRIP has not been merged and first and last TRIP_Id are in fact the same
    full_trips_concat_gdf['TRIP_ID_LAST'] = np.where(full_trips_concat_gdf.TRIP_ID_LAST.isnull(), full_trips_concat_gdf.TRIP_ID, full_trips_concat_gdf.TRIP_ID_LAST)


    return full_trips_concat_gdf.reset_index(drop=True), trip_concat_dict


def extract_concatenated_trips(full_trips_concat_gdf, gdf_sp, trip_sp_gdf, gdf_ep, trip_ep_gdf):
    # Filter for those trip_ids that are still the start of a trip even after the concatenation (of trip chains)
    t_id_sp = full_trips_concat_gdf.TRIP_ID_FIRST
    t_id_ep = full_trips_concat_gdf.TRIP_ID_LAST

    # Also filter dfs that contain points
    gdf_sp_concat = gdf_sp.query("TRIP_ID in @t_id_sp")
    trip_sp_gdf_concat = trip_sp_gdf.query("TRIP_ID in @t_id_sp")

    gdf_ep_concat = gdf_ep.query("TRIP_ID in @t_id_ep")
    trip_ep_gdf_concat = trip_ep_gdf.query("TRIP_ID in @t_id_ep")

    assert len(trip_sp_gdf_concat) == len(trip_ep_gdf_concat) == len(gdf_sp_concat) == len(gdf_ep_concat)

    return gdf_sp_concat, trip_sp_gdf_concat, gdf_ep_concat, trip_ep_gdf_concat

    
def getIndexInList(trip_id, full_trip_gdf):
    """This function takes in a trip_id and returns the list index of this trip's position in the ground truth clustering.

    Args:
        trip_id (int): TRIP_ID

    Returns:
        int: The index of this TRIP_ID in the ground truth clustering vector.
    """
    index_list = full_trip_gdf.sort_values('TRIP_ID').TRIP_ID.to_list()

    return index_list.index(trip_id)


def build_clustering_after_concatenation(full_trips_concat_gdf, trip_concat_dict, full_trip_gdf):
    """This function builds the clustering vector after the concatenation step.

    Args:
        full_trips_concat_gdf (GeoDataFrame): GeoDataFrame containing all trips after the concatenation step.
        trip_concat_dict (dict): Dictionary containing the trip chains that were concatenated.

    Returns:
        int: The index of this TRIP_ID in the ground truth clustering vector.
    """

    # This creates the array with clustering IDs after the concatenation step
    clustering_concat = {}
    for index, trip in full_trips_concat_gdf.reset_index().sort_values('TRIP_ID').iterrows():
        trip_order_index = getIndexInList(trip.TRIP_ID, full_trip_gdf)

        clustering_concat[trip_order_index] = index

        if trip.TRIP_ID in trip_concat_dict:
            for t in trip_concat_dict[trip.TRIP_ID]:
                clustering_concat[getIndexInList(t, full_trip_gdf)] = index

    clustering_concat = list(dict(sorted(clustering_concat.items())).values())

    print(f"Number of unique clusters: {len(set(clustering_concat))}")

    return clustering_concat


def build_hl_from_start_points(gdf_sp):
    # Generate home locations (HL) from SPs
    gdf_sp.index=pd.to_datetime(gdf_sp.TRIP_START)
    gdf_sp['hl'] = gdf_sp['TRIP_START'].apply(lambda x: 1 if x in gdf_sp.between_time('6:00', '10:00').TRIP_START else 0).astype(object)
    gdf_sp.reset_index(inplace=True, drop=True)

    # Extract only those cells that are HL
    gdf_hl_sp = gdf_sp[gdf_sp['hl'] == 1]

    # create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(gdf_hl_sp)

    # get component labels
    components = W.component_labels

    gdf_hl_combined_sp = pd.merge(gp.sjoin(
        gdf_hl_sp,
        gdf_hl_sp.dissolve(by=components)[["geometry"]],
        how="left"
    ), gdf_hl_sp.dissolve(by=components)[["geometry"]].reset_index(), left_on="index_right", right_on='index', suffixes=("__drop", "")).drop(['index', 'index_right', 'geometry__drop'], axis=1)

    # Compute count of unique HL per Peson (HL here is already the merged tiles)
    gdf_hl_combined_sp = pd.merge(gdf_hl_combined_sp, gdf_hl_combined_sp.astype({'geometry': 'string'}).groupby('PERSON_ID')[['geometry']].nunique().reset_index().rename(columns={'geometry': 'CNT_UNIQUE_HL'}), how="left")

    # using dictionary to convert specific columns
    convert_dict = {'PERSON_ID': object,
                    'CNT_UNIQUE_HL': int
                    }
    
    gdf_hl_combined_sp = gdf_hl_combined_sp.astype(convert_dict)

    return gdf_hl_combined_sp


def build_hl_from_end_points(gdf_ep):
    # Generate home locations (HL) from EPs
    gdf_ep.index=pd.to_datetime(gdf_ep.TRIP_END)
    gdf_ep['hl'] = gdf_ep['TRIP_END'].apply(lambda x: 1 if x in gdf_ep.between_time('18:00', '00:00').TRIP_END else 0).astype(object)
    gdf_ep.reset_index(inplace=True, drop=True)

    # Extract only those cells that are HL
    gdf_hl_ep = gdf_ep[gdf_ep['hl'] == 1]

    ### Merge hl cells that are adjacent (touching) to each other 
    # create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(gdf_hl_ep)

    # get component labels
    components = W.component_labels

    gdf_hl_combined_ep = pd.merge(gp.sjoin(
        gdf_hl_ep,
        gdf_hl_ep.dissolve(by=components)[["geometry"]],
        how="left"
    ), gdf_hl_ep.dissolve(by=components)[["geometry"]].reset_index(), left_on="index_right", right_on='index', suffixes=("__drop", "")).drop(['index', 'index_right', 'geometry__drop'], axis=1)

    gdf_hl_combined_ep = pd.merge(gdf_hl_combined_ep, gdf_hl_combined_ep.astype({'geometry': 'string'}).groupby('PERSON_ID')[['geometry']].nunique().reset_index().rename(columns={'geometry': 'CNT_UNIQUE_HL'}), how="left")

    # using dictionary to convert specific columns
    convert_dict = {'PERSON_ID': object,
                    'CNT_UNIQUE_HL': int
                    }
    
    gdf_hl_combined_ep = gdf_hl_combined_ep.astype(convert_dict)

    return gdf_hl_combined_ep

def concatenate_hl(gdf_hl_combined_sp, gdf_hl_combined_ep):
    gp_combined = pd.concat([gdf_hl_combined_ep, gdf_hl_combined_sp])

    ### Merge hl cells that are adjacent (touching) to each other 
    # create spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(gp_combined)

    # get component labels
    components = W.component_labels

    # We need to first join and then merge to first get the right index and then actually join the geometry 
    gp_combined = pd.merge(gp.sjoin(
        gp_combined,
        gp_combined.dissolve(by=components)[["geometry"]],
        how="left"
    ), gp_combined.dissolve(by=components)[["geometry"]].reset_index(), left_on="index_right", right_on='index', suffixes=("__drop", "")).drop(['index', 'index_right', 'geometry__drop'], axis=1)

    gp_combined = pd.merge(gp_combined.drop('CNT_UNIQUE_HL', axis=1), gp_combined.astype({'geometry': 'string'}).groupby('PERSON_ID')[['geometry']].nunique().reset_index().rename(columns={'geometry': 'CNT_UNIQUE_HL'}), how="left")

    # using dictionary to convert specific columns
    convert_dict = {'PERSON_ID': object,
                    'CNT_UNIQUE_HL': int
                    }
    
    gp_combined = gp_combined.astype(convert_dict)

    # print number of unique person IDs per HL count
    print('Count unique PERSON_IDs per HL: ',gp_combined.astype({'geometry':'string'}).groupby(['geometry'])[['PERSON_ID']].nunique().value_counts())

    # Number of users for which at least one HL was identified
    print('Number of users for which at least on Home Location has been identified: ', gp_combined.PERSON_ID.unique().size)

    # Assign ID to HL
    gp_combined['HL_ID'] = gp_combined.astype({'geometry': 'string'}).groupby('geometry').ngroup()

    HL_table = gp_combined[['geometry', 'HL_ID']].drop_duplicates()

    print(f"Number of unique HL tiles: {len(HL_table)}")

    return gp_combined, HL_table


def match_trips_to_HL(gp_combined, HL_table, trip_sp_gdf_concat, trip_ep_gdf_concat, full_trips_concat_gdf):
    # Merge all start and enpoints of all trajectories with HL tiles
    # All successfully matched trips will have 0 in the "matched_sp/ep" column else NaN
    matched_sp = gp.sjoin(
        trip_sp_gdf_concat, # This data frame contains all SPs of the trips that are at the end of a concatenated trip (end of a concatenated trip)
        gp_combined.dissolve()[['geometry']], # Here we do the matching on the dissolved HL tiles since we only want to have one match per point to detect binary whether it is matched at all or not
        how="left"
    ).rename(columns={"index_right": "matched_sp"})

    matched_ep = gp.sjoin(
        trip_ep_gdf_concat, # This data frame contains all EPs of the trips that are at the end of a concatenated trip (end of a concatenated trip)
        gp_combined.dissolve()[['geometry']], # same here, see above
        how="left"
    ).rename(columns={"index_right": "matched_ep"})


    # Merge start and endpoints of all trajectories with HL tiles to get HL_IDs for each trip
    # Note: Since we here match with the dissolved tile, we also can at max get ONE match per SP since overlapping HL tiles are dissolved.
    s = gp.sjoin(
        trip_sp_gdf_concat,
        HL_table,
        how="right").drop('index_left', axis=1).dropna()

    e = gp.sjoin(
        trip_ep_gdf_concat, 
        HL_table, 
        how="right").drop('index_left', axis=1).dropna()

    # Get unmatched start and endpoints
    unmatched_sp_t_ids = matched_sp[matched_sp.matched_sp.isnull()].TRIP_ID.to_list()
    unmatched_ep_t_ids = matched_ep[matched_ep.matched_ep.isnull()].TRIP_ID.to_list()

    # Number of unmatched trajectories that do not start or end in an HL tile
    nr_unmatched = len(full_trips_concat_gdf.query('TRIP_ID_FIRST in @unmatched_sp_t_ids and TRIP_ID_LAST in @unmatched_ep_t_ids'))
    print(f"Number of unmatched trajectories (concatenated) that do neither start nor end in a HL tile: {nr_unmatched}/{len(full_trips_concat_gdf)}")

    # Get TRIP_IDs of matched start and endpoints
    matched_sp_t_ids = matched_sp[~matched_sp.matched_sp.isnull()].TRIP_ID.to_list()
    matched_ep_t_ids = matched_ep[~matched_ep.matched_ep.isnull()].TRIP_ID.to_list()

    print(f"Number of trajectories (concatenated) that start AND end in a HL tile: {len(full_trips_concat_gdf.query('TRIP_ID_FIRST in @matched_sp_t_ids and TRIP_ID_LAST in @matched_ep_t_ids'))}/{len(full_trips_concat_gdf)}")

    # check whether number of unmatched trajectories plus number of matched trajectories do line up with the total number of trips in data (in this case concatenated trips)
    assert (full_trips_concat_gdf.query("TRIP_ID_FIRST in @s.TRIP_ID or TRIP_ID_LAST in @e.TRIP_ID").TRIP_ID.nunique() + nr_unmatched) == len(full_trips_concat_gdf)

    # Merge matched SP and EP to get the HL_IDs for each trip and drop duplicates.
    HL_table_se_concat = pd.merge(full_trips_concat_gdf, s[['TRIP_ID', 'HL_ID']], left_on="TRIP_ID_FIRST", right_on="TRIP_ID", how="left")
    HL_table_se_concat = pd.merge(HL_table_se_concat, e[['TRIP_ID', 'HL_ID']], left_on="TRIP_ID_LAST", right_on="TRIP_ID", how="left").drop(['TRIP_ID_y', 'TRIP_ID'], axis=1).rename(columns={'TRIP_ID_x': 'TRIP_ID'})
    HL_table_se_concat = HL_table_se_concat[['TRIP_ID', 'HL_ID_x', 'HL_ID_y']].set_index('TRIP_ID').stack().droplevel(1).reset_index().rename(columns={0: 'HL_ID'}).drop_duplicates()

    # Get trips that match different HL tiles with their SP and EP
    double_assigned_trips = HL_table_se_concat.groupby('TRIP_ID').filter(lambda x: len(x) > 1)

    # Get trips that are not assigned to any HL tile
    unmatched_trips = full_trips_concat_gdf.query('TRIP_ID_FIRST in @unmatched_sp_t_ids and TRIP_ID_LAST in @unmatched_ep_t_ids')[['TRIP_ID']].reset_index(drop=True)
    unmatched_trips['HL_ID'] = None # Get same format as HL_table_se_concat

    #HL_table_se_concat['HL_ID'] = HL_table_se_concat['HL_ID_x'].combine_first(HL_table_se_concat['HL_ID_y'])
    print(f"Number of trips that match different HL tiles with their SP and EP: {double_assigned_trips.TRIP_ID.nunique()}")

    return HL_table_se_concat, unmatched_trips, double_assigned_trips, nr_unmatched


def assign_double_matched_trips_to_unique_hl(HL_table_se_concat, full_trips_concat_gdf, unmatched_trips, double_assigned_trips, nr_unmatched):
    # Get trips that match only one HL tile with their SP and EP
    uniquely_assigned_trips = HL_table_se_concat.groupby('TRIP_ID').filter(lambda x: len(x) == 1)

    def compute_lcss_scores(double_assigned_trip):
        t_id = double_assigned_trip.TRIP_ID
        hl_id = double_assigned_trip.HL_ID

        # Create dict to store results for this trip and HL
        result_dict = {t_id: {hl_id: list()}}

        # Get trajectory linestring for this trip
        trip = full_trips_concat_gdf.query("TRIP_ID == @t_id")

        # Get trips that are uniquely assigned to this HL
        assigned_trips = uniquely_assigned_trips.query("HL_ID == @hl_id")

        # Loop through these trips and calc LCSS scores for each of them
        for index, assigned_trip in assigned_trips.iterrows():
            assigned_t_id = assigned_trip.TRIP_ID

            # Skip the calc for the trip with itself
            if assigned_t_id == t_id:
                continue
            # Get trajectory linestring for this trip (here we use the non-concated one since we are considering S and E points separately to match HL with the concated trips afterwards)
            a_trip = full_trips_concat_gdf.query("TRIP_ID == @assigned_t_id")

            score = LCSS(trip.geometry, a_trip.geometry)

            # save scores in list
            result_dict[t_id][hl_id].append(score)
        return result_dict

    parallel_scores = Parallel(n_jobs=-2, verbose=10)(delayed(compute_lcss_scores)(double_assigned_trip) for index, double_assigned_trip in double_assigned_trips.iterrows())

    # Flatten the list of dicts from parallel processing
    lcss_scores = {}
    for idx, trip in enumerate(parallel_scores):
        for t_id in trip:

            # create dict for this trip if not yet existing (it would if another HL this trip was joined with has already been checked)
            if t_id not in lcss_scores:
                lcss_scores[t_id] = {}

            for hl_id in parallel_scores[idx][t_id]:
                # create new list for this HL under the trip key
                lcss_scores[t_id][hl_id] = parallel_scores[idx][t_id][hl_id]

    # Get and compare max scores across all matched HL for a trip and assign the HL with the max value of any trip
    for trip in lcss_scores:
        for key in lcss_scores[trip]:
            if len(lcss_scores[trip][key]) > 0:
                lcss_scores[trip][key] = max(lcss_scores[trip][key])
            else:
                lcss_scores[trip][key] = 0
        
        lcss_scores[trip] = max(lcss_scores[trip], key=lcss_scores[trip].get)

    # Assign resolved scores to se_HL_lookup table and drop duplicates
    HL_table_se_concat['HL_ID'] = HL_table_se_concat.apply(lambda x: lcss_scores[x['TRIP_ID']] if x['TRIP_ID'] in lcss_scores else x['HL_ID'], axis=1)
    HL_table_se_concat = HL_table_se_concat.drop_duplicates(['TRIP_ID', 'HL_ID']).reset_index(drop=True)

    # Assert that the number of trips that are matched to a HL tile plus the the nr of trips that are unmatched is equal to the total number of concatenated trips
    assert len(HL_table_se_concat) + nr_unmatched == len(full_trips_concat_gdf)

    # Combine resolved HL HL_table_se_concat with the unmatched trips to get full HL_table for all trips
    HL_table_trips_concat = pd.concat([HL_table_se_concat, unmatched_trips], ignore_index=True)
    HL_table_trips_concat.loc[HL_table_trips_concat.HL_ID.isnull(), 'HL_ID'] = -1 # assign HL_ID -1 to unmatched trips


    return HL_table_trips_concat


def getTripOverlaps(gdf):
    """This function calculates whether two trips overlap in time.

    Args:
        gdf (_type_): The GeoDataFrame containing the trips.

    Returns:
        _type_: Dictionary with trip IDs as keys and a list of trip IDs that overlap with the key trip as values.
    """
    def getOverlaps(trip_x):
        overlap_dict = {}
        overlaps = []
        ts_x = pd.to_datetime(trip_x['TRIP_START'], format='%Y-%m-%d %H:%M:%S')
        te_x = pd.to_datetime(trip_x['TRIP_END'], format='%Y-%m-%d %H:%M:%S')
        i = 0

        for index_y, trip_y in gdf.iterrows():
            ts_y = pd.to_datetime(trip_y['TRIP_START'], format='%Y-%m-%d %H:%M:%S')
            te_y = pd.to_datetime(trip_y['TRIP_END'], format='%Y-%m-%d %H:%M:%S')

            if ts_x <= ts_y and te_x >= ts_y and (trip_x.TRIP_ID != trip_y.TRIP_ID): 
                overlaps.append(trip_y['TRIP_ID'])
            else:
                pass
            
        overlap_dict[trip_x['TRIP_ID']] = overlaps

        return overlap_dict

    overlap_dicts = Parallel(n_jobs=-2, verbose=10)(delayed(getOverlaps)(trip_x) for index_x, trip_x in gdf.iterrows())

    return {k: v for d in overlap_dicts for k, v in d.items()}


def getOverlappingTrips(traj_id_list, full_trips_concat_gdf_overlap_dict):
    """This function finds the overlapping trips for a list of trajectory IDs.

    Args:
        traj_id_list (_type_): List of trajectory IDs.
    """
    overlapping_trips = [item for sublist in [full_trips_concat_gdf_overlap_dict[t] for t in traj_id_list] for item in sublist] # we first get a list of lists and then flatten it
    return overlapping_trips

def findLargestNonSimultaneousSubset(traj_id_list, full_trips_concat_gdf_overlap_dict, RANDOMIZED_SEARCH_THRESHOLD=300, RANDOMIZED_SEARCH_ITERATIONS=1000):
    """This function finds the largest subset of trajectories that are not simultaneous. It uses a determinitic algorithm if the length of the trajectory ID list is smaller than a threshold and a randomized algorithm if the length is larger than the given threshold.

    Args:
        traj_id_list (list): List of trajectory IDs.

    Returns:
        list: List of trajectory IDs that are not simultaneous.
    """
    len_traj_id_list = len(traj_id_list)

    # Deterministic (optimal) algorithm
    if len_traj_id_list <= RANDOMIZED_SEARCH_THRESHOLD:
        # We create a list of all possible subsets of the trajectory ID list with decreasing length
        # We do this iteratively to not overload the memory
        for i in range(len_traj_id_list):
            # Create a list of all possible subsets of the trajectory ID list with length len_traj_id_list - i
            subsets = list(itertools.combinations(traj_id_list, len_traj_id_list - i))
        
            # Sort the list by length of the subsets
            subsets.sort(key=len, reverse=True)

            # Loop through the list of subsets
            for subset in subsets:
                # get all trips that do overlap in time with any of the trips in subset
                overlapping_trips = getOverlappingTrips(subset, full_trips_concat_gdf_overlap_dict)

                # Check if the subset is not simultaneous
                if all([t not in overlapping_trips for t in subset]):
                    # If so, return the subset as list
                    return list(subset)

    # Randomized algorithm (numeric approximation)
    else:
        def randomized_subset_search(traj_id_list):
            subsets = []
            candidates = traj_id_list.copy()
            id = candidates.pop(random.randrange(len(candidates)))
            subset = [id]
            while candidates:
                next = candidates.pop(random.randrange(len(candidates)))

                if all([t not in getOverlappingTrips(subset, full_trips_concat_gdf_overlap_dict) for t in subset]):
                    subset.append(next)
            
            subsets.append(subset)
            return max(subsets, key=len)
        
        # We run the randomized subset search 100 times and return the longest subset
        print(f'Running randomized subset for search for {RANDOMIZED_SEARCH_ITERATIONS} iterations with {len(traj_id_list)} trajectories...')
        result = Parallel(n_jobs=-2, verbose=10)(delayed(randomized_subset_search)(traj_id_list) for _ in range(RANDOMIZED_SEARCH_ITERATIONS))
        print('Done. Length of longest subset: ', len(max(result, key=len)))
        return max(result, key=len)


def build_clustering_after_HL_assignment(HL_table_trips_concat, full_trip_gdf, trip_concat_dict, full_trips_concat_gdf_overlap_dict):
    # This creates the array with clustering IDs after the HL assignment step
    clustering_after_HL = {}
    HL_table_dict = (HL_table_trips_concat.groupby('HL_ID')
        .apply(lambda x: list(dict(x.TRIP_ID).values()))
        .to_dict())

    for index, HL in tqdm(enumerate(HL_table_dict), total=len(HL_table_dict)):
        # Skip HL_ID -1, we will assign a clustering ID to these trips later
        if HL == -1:
            continue

        # find the largest subset of trips that are not simultaneous
        non_simultaneous_subset = findLargestNonSimultaneousSubset(HL_table_dict[HL], full_trips_concat_gdf_overlap_dict)

        # assign hl_id -1 to all trips that are not part of the largest subset
        for trip in HL_table_dict[HL]:
            if trip not in non_simultaneous_subset:
                HL_table_dict[-1].append(trip)

        # Loop through all trips that are assigned to this HL_ID and assign the same clustering ID to all of them
        for trip in non_simultaneous_subset:
            clustering_after_HL[getIndexInList(trip, full_trip_gdf)] = index

            # Check if this trip is a concatenated trip and assign the same clustering ID to all trips that are part of the concatenated trip
            if trip in trip_concat_dict:
                for t in trip_concat_dict[trip]:
                    clustering_after_HL[getIndexInList(t, full_trip_gdf)] = index

    # Assign clustering IDs to all hl_id -1 trips (these are the trips that were not successfully assigned to any HL_ID)
    for index, unm_trip in enumerate(HL_table_dict[-1]):
        clustering_after_HL[getIndexInList(unm_trip, full_trip_gdf)] = index + len(HL_table_dict) - 1 # -1 because we don't want to count the -1 HL_ID in the length of HL_table_dict

        # Check if this trip is a concatenated trip and assign the same clustering ID to all trips that are part of the concatenated trip
        if unm_trip in trip_concat_dict:
            for t in trip_concat_dict[unm_trip]:
                clustering_after_HL[getIndexInList(t, full_trip_gdf)] = index + len(HL_table_dict) - 1

    return clustering_after_HL, HL_table_dict



def assign_trips_without_match(clustering_after_HL, HL_table_dict, full_trips_concat_gdf, full_trips_concat_gdf_overlap_dict, full_trip_gdf, trip_concat_dict, SIM_THRESH_FOR_NO_MATCH=0.25):
    # This compares all trips that were not assigned to any HL_ID with all trips that were assigned to a HL_ID and assigns the clustering ID of the trip with the highest LCSS score above a certain threshold
    new_cluster_ids = []

    print('Comparing trips that were not assigned to any HL_ID with trips that were assigned to a HL_ID...')
    for unm_trip in tqdm(HL_table_dict[-1], total=len(HL_table_dict[-1])):
        # compute LCSS for all HL_IDs in clustering_after_HL
        scores = {}
        most_similar_trip_per_hl_id = {}
        for hl_id, trips in HL_table_dict.items():
            scores[hl_id] = {}
            for t in trips:
                if t != unm_trip: # don't compare trip with itself in case for -1 hl_id
                    scores[hl_id][t] = LCSS(full_trips_concat_gdf.query("TRIP_ID == @unm_trip").geometry, full_trips_concat_gdf.query("TRIP_ID == @t").geometry)

            # Store most similar trip per HL_ID
            most_similar_trip_per_hl_id[hl_id] = max(scores[hl_id], key=scores[hl_id].get)

            # find the trip with the highest LCSS score for current HL_ID
            max_score = max(scores[hl_id].values())
            scores[hl_id] = max_score
            
        # find the HL_ID with the highest LCSS score
        max_score_across_hl_id = max(scores.values())
        max_score_hl_id = [k for k, v in scores.items() if v == max_score_across_hl_id] #  in theory there could be two trips of different HL_ids with the same LCSS score. This is why this is a list

        # throw error if length of max_score_hl_id is greater than 1 and the max_score_across_hl_id is above the threshold
        if len(max_score_hl_id) > 1 and max_score_across_hl_id > SIM_THRESH_FOR_NO_MATCH:
            print(scores)
            print(unm_trip)
            print(most_similar_trip_per_hl_id[max_score_hl_id[0]])
            print(most_similar_trip_per_hl_id[max_score_hl_id[1]])
            print("There are two HL_IDs with the same LCSS score. Looking for second highest LCSS score...")

            # find the second highest LCSS score
            scores[max_score_hl_id[0]] = [LCSS(full_trips_concat_gdf.query("TRIP_ID == @unm_trip").geometry, full_trips_concat_gdf.query("TRIP_ID == @t").geometry) for t in HL_table_dict[max_score_hl_id[0]] if t != unm_trip]
            scores[max_score_hl_id[1]] = [LCSS(full_trips_concat_gdf.query("TRIP_ID == @unm_trip").geometry, full_trips_concat_gdf.query("TRIP_ID == @t").geometry) for t in HL_table_dict[max_score_hl_id[1]] if t != unm_trip]
            
            # find the HL_ID with the second highest LCSS score and use that as the max_score_hl_id
            if sorted(scores[max_score_hl_id[0]])[-2] > sorted(scores[max_score_hl_id[1]])[-2]:
                max_score_hl_id = max_score_hl_id[0]
            else:
                max_score_hl_id = max_score_hl_id[1]
            print("Done. Second highest LCSS score is for HL_ID", max_score_hl_id)

        else:
            max_score_hl_id = max_score_hl_id[0]

        # if the highest LCSS score is above the threshold and the trips that have the highest LCSS score do not overlap in time
        if max_score_across_hl_id > SIM_THRESH_FOR_NO_MATCH and unm_trip not in getOverlappingTrips(HL_table_dict[max_score_hl_id], full_trips_concat_gdf_overlap_dict): 
            # if hl_id of matched hl_id is -1 then create new cluster for the two trips
            if max_score_hl_id == -1:
                # Create new cluster id that i one higher than the highest clustering ID and the number of new clusters
                new_cluster_id = max(clustering_after_HL.values()) + 1 + len(new_cluster_ids)
                print("no match and assign new cluster id", new_cluster_id, "to trips", unm_trip, most_similar_trip_per_hl_id[max_score_hl_id])

                # create new cluster with ID that is one higher than the highest clustering ID
                HL_table_dict[new_cluster_id] = [unm_trip, most_similar_trip_per_hl_id[max_score_hl_id]]
                new_cluster_ids.append(new_cluster_id)

                # Remove trips from -1 hl_id to avoid two equal LCSS scores error
                HL_table_dict[-1].remove(most_similar_trip_per_hl_id[max_score_hl_id])
                HL_table_dict[-1].remove(unm_trip)

            # if hl_id of matched hl_id is not -1 then assign the clustering ID of the matched HL_ID to the trip
            else:
                print("existing match and assign cluster id", max_score_hl_id, "to trip", unm_trip)
                # Append the trip to the list of trips that are part of the HL_ID with the highest LCSS score
                HL_table_dict[max_score_hl_id].append(unm_trip)

                # Remove trip from -1 hl_id to avoid two equal LCSS scores error
                HL_table_dict[-1].remove(unm_trip)

                # assign the clustering ID of the HL_ID with the highest LCSS score to the trip
                # This overwrites the previous clustering ID of the trip that was assigned in the previous step
                clustering_after_HL[getIndexInList(unm_trip, full_trip_gdf)] = clustering_after_HL[getIndexInList(HL_table_dict[max_score_hl_id][0], full_trip_gdf)] # This is a list of length one as asserted above

                # Check if this trip is a concatenated trip and assign the same clustering ID to all trips that are part of the concatenated trip
                if unm_trip in trip_concat_dict:
                    for t in trip_concat_dict[unm_trip]:
                        clustering_after_HL[getIndexInList(t, full_trip_gdf)] = clustering_after_HL[getIndexInList(HL_table_dict[max_score_hl_id][0], full_trip_gdf)] # This is a list of length one as asserted above
        else:
            # if trip overlaps in time with other trips in the same HL_ID
            if unm_trip in getOverlappingTrips(HL_table_dict[max_score_hl_id], full_trips_concat_gdf_overlap_dict):
                print("trip overlaps in time with other trips in the same HL_ID")
    print('Done.')        

    # Assign clustering IDs to all trips that are part of a new cluster
    print('Assigning clustering IDs to all trips that are part of a new cluster...')
    for cluster_id in new_cluster_ids:
        for trip in HL_table_dict[cluster_id]:
            clustering_after_HL[getIndexInList(trip, full_trip_gdf)] = cluster_id

        # Assign clustering IDs to all trips that are part of a concatenated trip
        if trip in trip_concat_dict:
            for t in trip_concat_dict[trip]:
                clustering_after_HL[getIndexInList(t, full_trip_gdf)] = cluster_id
    print('Done.')

    return clustering_after_HL






