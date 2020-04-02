#!/bin/bash

# to sort by 2nd column in csv file (character - ',' after 't' for delimiter in each line)
sort -t, -nk2 training_object_detections_count.csv > training_sorted_detections_count.csv

# to delete first line (d for delete and 1 for first line)
sed 1d training_sorted_detections_count.csv > temp.csv

# to add a column at beginning with row numbers in csv file
awk '{print NR "," $0}' temp.csv > temp1.csv

# to delete 2nd column in csv file (d for delete and the character - ',' after 'd' for delimiter in each line)
cut -d, -f2 --complement temp1.csv > training_indexed_sorted_detections_count.csv

# to print last line in file
awk 'END { print }' training_indexed_sorted_detections_count.csv

# command to execute eda_test.gnu script
gnuplot -p eda_test.gnu

# to get mean of 2nd column
awk -F ',' '{x+=$2}END{print "MEAN: " x/NR}' training_indexed_sorted_detections_count.csv

# to get standard deviation of 2nd columns
awk -F ',' '{x+=$2;y+=$2^2}END{print "STD: " sqrt(y/NR-(x/NR)^2)}' training_indexed_sorted_detections_count.csv
