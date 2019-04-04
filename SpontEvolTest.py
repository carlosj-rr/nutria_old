import numpy as np
import scipy
from inputfile import * # Must add a check the the file exists, and that the
			# variables are valid
from numpy import exp
from scipy import stats

class Offspring(object):
	def __init__(self,name,parent):
		self.name = name
		self.parent = parent



class NewOrganism(object):
	#Constructor
	def __init__(self,name,number_of_genes,dev_steps):
		self.name = name
		self.generation = 0
		self.number_of_genes = number_of_genes
		self.prop_unlinked = prop_unlinked
		self.prop_no_threshold = prop_no_threshold
		self.thresh_min = thresh_min
		self.thresh_max = thresh_max
		self.decay_min = decay_min
		self.decay_max = decay_max
		self.dev_steps = dev_steps
		self.decays = decays = randomMaskedVector(number_of_genes,0,decay_min,decay_max)
		self.thresholds = thresholds = randomMaskedVector(number_of_genes,prop_no_threshold,thresh_min,thresh_max)
		self.start_vect = start_vect = makeStartVect(number_of_genes)
		self.grn = grn = makeGRN(number_of_genes,prop_unlinked)
		self.development = development = develop(start_vect,grn,decays,thresholds,dev_steps)
		self.fitness = fitness = calcFitness(development)
		self.sequences = sequences = makeRandomSequenceArray(seq_length,base_props,number_of_genes) if fitness != 0 else None


""" The population class below is of unrelated GRNs. This is only useful
when trying to find a founder. TO DO: add a way to construct population
based on a parent.start

To access attributes from the NewOrganism class, do
smth along the lines of:

>>> founder_pop.individuals[0].grn

Also, if you'd like to access a certain attribute of NewOrganism for all
Organisms in the population (here, the "name"), use something like:

>>> np.array([x.name for x in founder_pop.individuals])

"""
class NewPopulation(object):
	def __init__(self,pop_size):
		self.pop_size = pop_size
		self.individuals = individuals = producePop(pop_size)


def producePop(pop_size):
	pop = np.ndarray((pop_size,),dtype=np.object)
	for i in range(pop_size):
		name = "Org" + str(i)
		pop[i] = NewOrganism(name,num_genes,dev_steps)
	return(pop)

def makeGRN(numGenes,prop_unlinked):
	grn = randomMaskedVector(numGenes ** 2,prop_unlinked,-2,2)
	grn = grn.reshape(numGenes,numGenes)
	return(grn)

def makeRandomSequence(seq_length,base_props):
	bases = ("T","C","A","G")
	sequence = np.random.choice(bases,seq_length,p=base_props)
	return(sequence)

def makeRandomSequenceArray(seq_length,base_props,num_genes):
	vect_length = seq_length * num_genes
	seq_vect = makeRandomSequence(vect_length,base_props)
	seq_arr = seq_vect.reshape(num_genes,seq_length)
	return(seq_arr)

def makeStartVect(numGenes):
	startingVect = np.array([1] * 1 + [0] * (numGenes - 1))
	return(startingVect)

def mutateBase(base):				# This mutation function is equivalent to
	Bases = ("T","C","A","G")		# the JC model of sequence evolution
	change = [x for x in Bases if x != base]
	new_base = np.random.choice(change)
	return(new_base)

def mutateGRN(grn,mutation_rate,mutation_bounds,change_rate,change_bounds): # Func also used for thresholds + decays
	original_shape = grn.shape
	flat_grn = grn.flatten()
	active_links = np.array([i for i,x in enumerate(flat_grn) if x != 0])
	to_mutate = np.random.choice((0,1),active_links.size,p=(1-mutation_rate,mutation_rate))
	if sum(to_mutate):
		mutants_indexes = np.array([i for i,x in enumerate(to_mutate) if x == 1])
		mutants_grn_indexes = active_links[mutants_indexes]
		flat_grn[mutants_grn_indexes] = mutateLink(flat_grn[mutants_grn_indexes],mutation_bounds)
	else:
		None
	if change_rate != 0:
		to_change = np.random.choice((0,1),flat_grn.size,p=(1-change_rate,change_rate))
		#print("Selected",sum(to_change),"GRN links for changing")
		if sum(to_change):
			#print(sum(to_change),"is nonzero")
			changed_grn_indexes = np.array([i for i,x in enumerate(to_change) if x == 1])
			#print("Indexes of links to be changed:",changed_grn_indexes)
			min_val,max_val = change_bounds
			if sum(to_change) > 1:
				#print(sum(to_change),"is greater than 1")
				flat_grn[changed_grn_indexes] = changeGRNLink_vect(flat_grn[changed_grn_indexes],min_val,max_val)
			else:
				#print(sum(to_change),"is 1")
				flat_grn[changed_grn_indexes] = changeGRNLink(flat_grn[changed_grn_indexes],min_val,max_val)
		else:
			#print("No sites selected for changing...maybe next time!")
			None
	else:
		#print("This system mutates but does not change")
		None
	grn = flat_grn.reshape(original_shape)
	return(grn)

def changeGRNLink(link_value,min_val,max_val):
		#print("Link with value",link_value,"will be changed")
		if link_value:
			#print("Value is nonzero...changing to zero")
			new_value = 0
		else:
			#print("Value is zero...changing it into a number between",min_val,max_val)
			range_size = max_val - min_val
			new_value = range_size * np.random.random() + min_val
		return(new_value)

changeGRNLink_vect = np.vectorize(changeGRNLink)

def mutateLink(link_value,link_mutation_bounds): # If a numpy array is passed, it's vectorized
	min_val,max_val = min(link_mutation_bounds),max(link_mutation_bounds)
	range_size = max_val-min_val
	result = link_value + range_size * np.random.random() + min_val
	return(result)

def mutateGenome(genome,seq_mutation_rate):
	original_dimensions = genome.shape
	flat_seq = genome.flatten()
	genome_length = len(flat_seq)
	ones_to_mutate = np.random.choice((0,1),genome_length,p=(1-seq_mutation_rate,seq_mutation_rate))
	if sum(ones_to_mutate):
		mutated_nucs = [i for i,x in enumerate(ones_to_mutate) if x ==1 ]
		for i in mutated_nucs:
			flat_seq[i] = mutateBase(flat_seq[i])
	final_seq = flat_seq.reshape(original_dimensions)
	return(final_seq)

def exponentialDecay(in_value,lambda_value):
	decayed_value = in_value * np.exp(-lambda_value)
	return(decayed_value)

def de_negativize(invect):
	true_falser = invect >= 0
	outvect = (invect * true_falser) + 0
	return(outvect)

def develop(startingVect,grn,decays,thresholds,developmentSteps):
	geneExpressionProfile = np.array([startingVect])
	#Running the organism's development, and outputting the results
	#in an array called geneExpressionProfile
	invect = startingVect
	for i in range(developmentSteps):
		decayed_invect = exponentialDecay(invect,decays)
		currV = grn.dot(invect) - thresholds	      # Here the threshold is subtracted.
		currV = de_negativize(currV) + decayed_invect # Think about how the thresholds
							      # should affect gene expression
		geneExpressionProfile = np.append(geneExpressionProfile,[currV],axis=0)
		invect = currV
	return(geneExpressionProfile)

def randomBinVect(totalInteractions,noInteractions):
	binVect = np.array([0] * noInteractions + [1] * (totalInteractions-noInteractions))
	np.random.shuffle(binVect)
	return(binVect)

def randomMaskedVector(numVals,propZero=0,minVal=0,maxVal=1):
	if minVal > maxVal:
		print("Error: minimum value greater than maximum value")
		return
	rangeSize = maxVal - minVal
	if propZero == 0:
		rpv = np.array(rangeSize * np.random.random(numVals) + minVal)
	else:
		numZeroes = round(propZero * numVals)
		mask = randomBinVect(numVals,numZeroes)
		rpv = np.array(rangeSize * np.random.random(numVals) + minVal)
		rpv = (rpv * mask) + 0
	return(rpv)

###### Functions for determining fitness
# (BELOW) Is the last gene ever expressed at a certain level, and also expressed in the last step?
def lastGeneExpressed(development,min_reproducin):
	dev_steps, num_genes = development.shape
	last_col_bool = development[:,(num_genes - 1)] > min_reproducin
	last_val_last_col = development[dev_steps - 1, (num_genes - 1)]
	if last_col_bool.any() and last_val_last_col > 0:
		return_val = True
	else:
		return_val = False
	return(return_val)

def propGenesOn(development):
	genes_on = development.sum(axis=0) > 0
	return(genes_on.mean())

def expressionStability(development):			# I haven't thought deeply about this.
	row_sums = development.sum(axis=1)		# What proportion of the data range is
	stab_val = row_sums.std() / (row_sums.max() - row_sums.min()) # the stdev? Less = better
	return(stab_val)

def exponentialSimilarity(development):
	dev_steps, num_genes = development.shape
	row_means = development.mean(axis=1)
	tot_dev_steps = dev_steps
	fitted_line = scipy.stats.linregress(range(tot_dev_steps),np.log(row_means))
	r_squared = fitted_line.rvalue ** 2
	return(r_squared)

def calcFitness(development):
	is_alive = lastGeneExpressed(development,min_reproducin)
	genes_on = propGenesOn(development)
	exp_stab = expressionStability(development)
	sim_to_exp = exponentialSimilarity(development)
	fitness_val = is_alive * np.mean([genes_on,exp_stab,sim_to_exp])
	return(fitness_val)

######

def mutateOrganism(organism):
	organism.generation = organism.generation + 1
	organism.grn = mutateGRN(organism.grn,grn_mutation_rate,link_mutation_bounds,prob_grn_change,new_link_bounds)
	organism.decays = de_negativize(mutateGRN(organism.decays,decay_mutation_rate,thresh_decay_mut_bounds,0,None))
	organism.thresholds = de_negativize(mutateGRN(organism.thresholds,thresh_mutation_rate,thresh_decay_mut_bounds,prob_thresh_change,thresh_decay_mut_bounds))
	organism.development = develop(organism.start_vect,organism.grn,organism.decays,organism.thresholds,organism.dev_steps)
	organism.fitness = fitness = calcFitness(organism.development)
	if organism.fitness != 0:
		organism.sequences = mutateGenome(organism.sequences,seq_mutation_rate)
	else:
		None
	

