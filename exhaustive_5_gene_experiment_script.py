#!/usr/bin/python

import numpy as np
import nutria as nt
import number_base_changer as bch

def run_exhaustive_study(num_genes=3,min_max_numbers=(-2,2),stepsize=0.2):
	min_num,max_num=min_max_numbers
	tot_distance=abs(min_num-max_num)
	num_steps=tot_distance/stepsize
	if num_steps.is_integer():
		new_number_base=int(num_steps + 1)
		tot_digits=num_genes**2
		max_number=str(int(num_steps))*tot_digits
		step_numbers=bch.convert_to_base10(int(max_number),new_number_base)
		print("This analysis will do",step_numbers,"iterations.")
		print(new_number_base,tot_digits,max_number)
	else:
		print("The range provided:",min_max_numbers,"cannot be divided into a whole number of steps of size",stepsize)
		return(1)

def non_exhaustive_study(num_genes,prop_unlinked,dev_steps,replicates):
	count=0
	rownum=0
	out_array=np.repeat(0.0,replicates*(num_genes**2+1)).reshape(replicates,num_genes**2+1)
	start_vect=nt.makeStartVect(num_genes)
	decays=np.repeat(0.1,num_genes)
	thresholds=np.repeat(0,num_genes)
	while count < replicates:
		grn=nt.makeGRN(num_genes,prop_unlinked)
#		grn=np.reshape(np.random.random(num_genes**2)*distance-maximum,(num_genes,num_genes))
		development=nt.develop(start_vect,grn,decays,thresholds,dev_steps,nonsenses = 1)
		fitness=nt.calcFitness(development)
		if fitness > 0:
			in_row=np.hstack((fitness,grn.flatten()))
			out_array[rownum,]=in_row
			rownum+=1
		count+=1
	outputname="output_"+str(replicates)+"reps.csv"
	np.savetxt(outputname,out_array[0:rownum],fmt="%5.5f",delimiter=",")
