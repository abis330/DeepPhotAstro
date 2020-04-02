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
