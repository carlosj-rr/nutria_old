#!/bin/bash
# DID NOT WORK. There isn't enough memory to loop over such a big number.
# Script to produce a csv file with an exhaustive list of parameters for the life simulations.

num_genes=$1
num_steps=$2
min_num=$3
max_num=$4

base=$(bc <<< $num_steps+1)
matrix_cells=$(bc <<< $num_genes^2)
total_permutations=$(bc <<< $base^$matrix_cells-1)

echo "Creating $total_permutations different states"

echo $(printf ',%.0s' $(seq 1 $(($matrix_cells-1)))) > output.csv

for i in $(seq 1 $total_permutations)
do
  echo $(bc <<< $i/$total_permutations*100)% done
  num_based_in_csv=$(bc <<< "obase=21;$i" | tr " " "," | sed 's/^,//g')
  num_digits=$(echo $num_based_in_csv | sed 's/,/"\n"/g' | wc -l)
  num_leading_zeroes=$(bc <<< $matrix_cells-$num_digits)
  leaders=$(printf '00,%.0s' $(seq 1 $num_leading_zeroes))
  final_num=$(echo "$leaders""$num_based_in_csv")
  echo $final_num >> output.csv
done
