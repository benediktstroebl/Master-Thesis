import csv
from itertools import groupby


## set variables ##
input_file_path = "input.csv"
output_file_path = "output.csv"
SAMPLE_RATE = 40
TID_COLUMN = 0
LNG_COLUMN = 1
LAT_COLUMN = 2
###################

with open(input_file_path, "rt") as fin:
    cr = csv.reader(fin)
    filecontents = [line for line in cr]

trips = {key: [",".join((point[LNG_COLUMN], point[LAT_COLUMN])) for point in list(trip)] \
    for key, trip in groupby(filecontents[1:], key=lambda x: x[TID_COLUMN])}

trips = [list(trip) for _, trip in groupby(filecontents[1:], key=lambda x: x[TID_COLUMN])]
trips = [trip[i] for trip in trips for i in range(0, len(trip)) \
         if ((i % SAMPLE_RATE == 0) | (i == len(trip)-1))] # add according to sample rate and last point

# TODO: also add last point

print(f"write sampled dataset to file")
with open(output_file_path, 'w') as csvfile: 
    write = csv.writer(csvfile)
    write.writerow(filecontents[0])
    write.writerows(trips)    