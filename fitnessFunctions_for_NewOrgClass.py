###### Functions for determining fitness
def lastGeneExpressed(organism,min_reproducin): # Is the last gene ever expressed at a certain
	development = organism.development	# level, and also expressed in the last step?
	last_col_bool = development[:,(organism.number_of_genes - 1)] > min_reproducin
	last_val_last_col = development[organism.dev_steps, (organism.number_of_genes - 1)]
	if last_col_bool.any() and last_val_last_col > 0:
		return_val = True
	else:
		return_val = False
	return(return_val)

def propGenesOn(organism):
	genes_on = organism.development.sum(axis=0) > 0
	return(genes_on.mean())

def expressionStability(organism):			# I haven't thought deeply about this.
	row_sums = organism.development.sum(axis=1)	# What proportion of the data range is
	stab_val = row_sums.std() / (row_sums.max() - row_sums.min()) # the stdev? Less = better
	return(stab_val)

def exponentialSimilarity(organism):
	row_means = organism.development.mean(axis=1)
	tot_dev_steps = dev_steps + 1 # (To include t=0)
	fitted_line = scipy.stats.linregress(range(tot_dev_steps),np.log(row_means))
	r_squared = fitted_line.rvalue ** 2
	return(r_squared)

def calcFitness(organism):
	is_alive = lastGeneExpressed(organism,min_reproducin)
	genes_on = propGenesOn(organism)
	exp_stab = expressionStability(organism)
	sim_to_exp = exponentialSimilarity(organism)
	fitness_val = is_alive * np.mean([genes_on,exp_stab,sim_to_exp])
	return(fitness_val)

