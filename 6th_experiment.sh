# In total, I have 200 alignments:
# 10 genes from the founder system, evolved 10 independent times
# starting from the same initial sequences. That's the first 100.
# The founder system is a randomly-generated GRN

# Then, 10 genes from the oner system, evolved 10 independent times
# starting from the same initial sequences (different from the founder system).
# That's the second 100. The oner system is a GRN without structure
# (see details in "oner_grn.csv")

# I'll do tree inferences for each alignment, then see how much the branch lengths vary
# through the permutations, on both cases.

for i in $(ls *.fas)
do
  iqtree -s $i -m JC
done

mkdir tmptrees
mkdir iqtreestuff
cp *treefile tmptrees/
mv *.fas.* iqtreestuff
cd tmptrees

echo "Gene,Rep,Sp1,Sp2,Sp3,Sp4,Stem" > 6th_exp_tabulated_blengths_Founder.csv
for i in $(seq 1 10)
do
  for j in $(seq 0 9)
  do
    lengths=$(cat Founder_a3456_rep"$j"_gene"$i".fas.treefile | sed 's/[(),_]*//g;s/gene[0-9]*//g;s/sp[0-9]//g;s/:/,/g;s/^,//g;s/;$//g')
    echo "$i","$j","$lengths" >> 6th_exp_tabulated_blengths_Founder.csv
  done
done

echo "Gene,Rep,Sp1,Sp2,Sp3,Sp4,Stem" > 6th_exp_tabulated_blengths_Oner.csv
for i in $(seq 1 10)
do
  for j in $(seq 0 9)
  do
    lengths=$(cat Oner_ao3456_rep"$j"_gene"$i".fas.treefile | sed 's/[(),_]*//g;s/gene[0-9]*//g;s/sp[0-9]//g;s/:/,/g;s/^,//g;s/;$//g')
    echo "$i","$j","$lengths" >> 6th_exp_tabulated_blengths_Oner.csv
  done
done

# I then combined both tables into a single one called 6th_exp_tabulated_blengths_all.csv, to be analyzed in R
