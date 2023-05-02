<p align="center">
<b><a href="#Abstract">Abstract</a></b>
|
<b><a href="#Methodology">Methodology</a></b>
|
<b><a href="#Conclusion">Results and Policy Implications</a></b>
|
<b><a href="#Disclaimer">Disclaimer</a></b>
</p>


# Estimating the success of re-identification in GPS trajectory datasets without user identifiers

This is the corresponding repository to my Master Thesis **Estimating the success of re-identification in GPS trajectory datasets without user identifiers**. It does not contain any data files due to privacy reasons.

## Abstract

With the advent of location-based services, vast amounts of individual-level mobility data are being generated and used for a variety of applications. These data can provide valuable insights for transport planning, urban development, and research. However, they also raise privacy concerns, as the high degree of uniqueness of human mobility traces can enable adversaries to link data back to individuals even without obvious identifiers. In this thesis, I propose a novel attack to reconstruct user identifiers in GPS trajectory datasets with no user-trajectory link. Specifically, I frame the problem of trajectory-user linking as an attack and evaluate the remaining privacy risk for users in such datasets. I also assess the efficacy of truncation as a simple privacy mechanism used in practice, to evaluate whether the protection it offers is sufficient. The findings show that the risk of re-identification is significant even when personal identifiers have been removed, and that simple privacy mechanisms may not be effective in protecting user privacy. This work highlights the need for a more holistic approach to privacy in location-based services and demonstrates the importance of evaluating the privacy risks of data-sharing practices.

## Methodology

### Data

In this study, I consider two mobility datasets containing GPS trajectories, namely the widely-used [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/) and a new non-pubic dataset collected by the *DLR MovingLab* and the researchers of the [freemove](https://www.freemove.space/en/) project. While these datasets contain other variables, this study only includes user ID, latitude, longitude, and timestamp information, with all additional information being discarded.

I apply the identical preprocessing steps to both datasets. To account for
measurement errors and artifacts in the data, I discard all trajectories that are
shorter than 200 meters or have less than 50 GPS recordings. This removes data
points that are the result of users accidentally starting to record or GPS connec-
tion issues that produced trips with only a few GPS points. Further, I remove the
longest 5% of all trips in both datasets to account for extreme outliers that are
likely the result of users not ending the recording after a trip has been completed.
Furthermore, analysis is restricted to trajectories that begin and end inside the
area of the bounding boxes depicted in Figures A.1 and A.2. These boxes contain
the urban areas of Berlin and Beijing and include a generous buffer around the
suburbs in order to mainly filter long-distance trips, e.g., flights, hence exclud-
ing only trajectories outside of the main city areas. This yields a total of 1,186
trajectories from 74 users and 5,101 trajectories from 73 users for freemove and
GeoLife, respectively.

You can explore the code for preprocessing, tessellation generation, descriptives, and loading the filtered data in these files:
- **Preprocessing**: [freemove](freemove/read_freemove.ipynb), [GeoLife](Geolife/read_geolife.ipynb)
- **Tessellation Generation**: [Berlin](freemove/generate_tessellation.ipynb), [Beijing](Geolife/generate_tessellation_geolife.ipynb)
- [Data Loaders](data_loader.py)
- **Descriptives**: [freemove](freemove/freemove_descriptives.ipynb), [GeoLife](Geolife/geolife_descriptives.ipynb)

The GeoLife dataset can downloaded via this [link](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/). The freemove dataset is not public.

### Attack

In this work I describe a technique to reconstruct user identifiers in GPS mobility datasets with no trajectory-user link by combining a set of assumptions about the day-to-day mobility patterns of urban residents. These include combining trips that seem to be continuations of one another and identifying potential home locations (HLs) of individuals. In addition, I leverage the information content of location co-visits to combine initially identified groups of trajectories and refine the cluster assignments.

An overview of the HL assignment step can be see from this flowchart.

<img src="https://github.com/benediktstroebl/Master-Thesis/blob/main/Figures/flowchart_HL.png" width="800px" align="center" />

During this procedure I use the Longest Common Subsequence (LCSS) to identifiy similar trajectories. I use a the maximum of the raw and reversed metric as illustrated in the following.

<img src="Figures/lcss.png" width="400px" align="center" />

Additionally, during evaluation, I apply a obfuscation technique to the raw data in order to assess the efficacy of this measure against the introduced attack. The following illustration explains the truncation method that is applied.

<img src="Figures/privacymethod.png" width="400px" align="center" />

You can explore the code for running the attack, the individual attack steps, and for obfuscating the data in these files:
- **Running the attack**: [freemove](freemove/read_freemove.ipynb), [GeoLife](Geolife/read_geolife.ipynb)
- **Individual methods used during the attack**: [Berlin](freemove/generate_tessellation.ipynb), [Beijing](Geolife/generate_tessellation_geolife.ipynb)
- [Data Loaders](data_loader.py)
- **Obfuscation of data**: [freemove](freemove/freemove_descriptives.ipynb), [GeoLife](Geolife/geolife_descriptives.ipynb)

### Evaluation

## Conclusion

for future algorithm development.

## Disclaimer

I have used GitHub co-pilot during implementation of the outlined research and relied on it for suggesting code writing comments and plotting. 
