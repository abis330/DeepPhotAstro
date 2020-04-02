# gnu script to plot detections count
set terminal pngcairo font "arial,10" size 1430,890
set datafile separator ','
set output 'barchart.png'
set style fill solid
set title "Number of Detections per Object"
set xlabel "Object ID"
set ylabel "Number of Detections"
set autoscale
# 1 is 1st column as x-axis and 2 is 2nd column for y-axis
plot "training_indexed_sorted_detections_count.csv" using 1:2 with dots 
unset output
