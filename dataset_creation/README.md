# Dataset Creation

This folder contains many files instrumental in preparing training and evaluation datasets as well as geocell and cluster generation.

obtain_data_scripts/ folder:
extract_adm1_geometries.py: extracts the admin level 1 geometries(states/provinces) from the gadm dataset and stores that file in data/gadm/countries_adm1_geometries.gpkd

filter_adm1_regions.py: filters out any regions that are not supported by google street view (the way we do this is by generating 20 samples per region, if none of them find street view imagery, then we delete them) and stores this in a file data/gadm/filtered_adm1_geometries.gpkg

calculate_region_samples.py: out of the filtered regions, calculate the number of total data samples per region (proportional to area of the region), save this in a csv file dataset_creation/adm1_sample_counts.csv

generate_coordinates.py: for each region, generate the corresponding number of samples.