# PROLOGUE. IMPORTING STUFF

import numpy as np
import scipy
import copy
from inputfile import * # Must add a check the the file exists, and that the
			# variables are valid
from numpy import exp
from scipy import stats


# CHAPTER 1. CLASS DECLARATIONS

##### ----- #####
class Organism(object):
	""" To access attributes from the Organism class, do
	smth along the lines of:
	>>> founder_pop.individuals[0].grn"""
	def __init__(self,name,generation,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences):
		self.name = name
		self.generation = generation
		self.num_genes = num_genes
		self.prop_unlinked = prop_unlinked
		self.prop_no_threshold = prop_no_threshold
		self.thresh_boundaries = thresh_boundaries
		self.decay_boundaries = decay_boundaries
		self.dev_steps = dev_steps
		self.decays = decays
		self.thresholds = thresholds
		self.start_vect = start_vect
		self.grn = grn
		self.development = development
		self.fitness = fitness
		self.sequences = sequences

##### ----- #####
class Population(object):
	"""To access a certain attribute of Organism for all
	Organisms in the population (for example, the "name"), use something like:
	>>> np.array([x.name for x in founder_pop.individuals])"""
	def __init__(self,pop_size,parent=None):
		self.pop_size = pop_size
		self.individuals = individuals = producePop(pop_size,parent)


# CHAPTER 2. MAIN FUNCTIONS TO CREATE AN ORGANISM, AND A POPULATION
# -- from scratch, or as a next generation

##### ----- #####
def makeNewOrganism(parent=None):
	if parent: 		# add also: if type(parent) is Organism:
		generation = parent.generation + 1
		name = parent.name.split("gen")[0] + "gen" + str(generation)
		decays = de_negativize(mutateGRN(parent.decays,decay_mutation_rate,thresh_decay_mut_bounds,0,None))
		thresholds = de_negativize(mutateGRN(parent.thresholds,thresh_mutation_rate,thresh_decay_mut_bounds,prob_thresh_change,thresh_decay_mut_bounds))
		start_vect = parent.start_vect
		grn = mutateGRN(parent.grn,grn_mutation_rate,link_mutation_bounds,prob_grn_change,new_link_bounds)
		development = develop(start_vect,grn,decays,thresholds,parent.dev_steps)
		fitness = calcFitness(development)
		if fitness == 0:
			sequences = None
		else:
			sequences = mutateGenome(parent.sequences,seq_mutation_rate)
		out_org = Organism(name,generation,parent.num_genes,parent.prop_unlinked,parent.prop_no_threshold,parent.thresh_boundaries,parent.decay_boundaries,parent.dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences)	
	else:
		name = "Lin" + str(int(np.random.random() * 1000000)) + "gen0"
		decays = randomMaskedVector(num_genes,0,decay_boundaries[0],decay_boundaries[1])
		thresholds = randomMaskedVector(num_genes,prop_no_threshold,thresh_boundaries[0],thresh_boundaries[1])
		start_vect = makeStartVect(num_genes)
		grn = makeGRN(num_genes,prop_unlinked)
		development = develop(start_vect,grn,decays,thresholds,dev_steps)
		fitness = calcFitness(development)
		if fitness == 0:
			sequences = None
		else:
			sequences = makeRandomSequenceArray(seq_length,base_props,num_genes)
		out_org = Organism(name,0,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences)
	return(out_org)

##### ----- #####
def producePop(pop_size,parent=None):
	pop = np.ndarray((pop_size,),dtype=np.object)
	if not parent:
		for i in range(pop_size):
			pop[i] = makeNewOrganism()
	else:
		if type(parent) is Organism:
			for i in range(pop_size):
				pop[i] = makeNewOrganism(parent)
		else:
			print("The type of the parent is not correct",type(parent))
	return(pop)


# CHAPTER 3. FUNCTIONS FOR MAKING AND ORGANISM FROM SCRATCH

########## ----- GRN RELATED ----- ##########
def makeGRN(numGenes,prop_unlinked):
	grn = randomMaskedVector(numGenes ** 2,prop_unlinked,-2,2)
	grn = grn.reshape(numGenes,numGenes)
	return(grn)

########## ----- SEQUENCE RELATED ----- ##########
def makeRandomSequence(seq_length,base_props):
	bases = ("T","C","A","G")
	sequence = np.random.choice(bases,seq_length,p=base_props)
	return(sequence)

##### ----- #####
def makeRandomSequenceArray(seq_length,base_props,num_genes):
	vect_length = seq_length * num_genes
	seq_vect = makeRandomSequence(vect_length,base_props)
	seq_arr = seq_vect.reshape(num_genes,seq_length)
	return(seq_arr)

########## ----- OTHER----- ##########
def makeStartVect(numGenes):
	startingVect = np.array([1] * 1 + [0] * (numGenes - 1))
	return(startingVect)

##### ----- #####
def exponentialDecay(in_value,lambda_value):
	decayed_value = in_value * np.exp(-lambda_value)
	return(decayed_value)

##### ----- #####
def de_negativize(invect):
	true_falser = invect >= 0
	outvect = (invect * true_falser) + 0
	return(outvect)

##### ----- #####
def randomBinVect(totalInteractions,noInteractions):
	binVect = np.array([0] * noInteractions + [1] * (totalInteractions-noInteractions))
	np.random.shuffle(binVect)
	return(binVect)

##### ----- #####
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


# CHAPTER 4. MUTATION FUNCTIONS

########## ----- GRN RELATED ----- ##########
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
		if sum(to_change):
			changed_grn_indexes = np.array([i for i,x in enumerate(to_change) if x == 1])
			min_val,max_val = change_bounds
			if sum(to_change) > 1:
				flat_grn[changed_grn_indexes] = changeGRNLink_vect(flat_grn[changed_grn_indexes],min_val,max_val)
			else:
				flat_grn[changed_grn_indexes] = changeGRNLink(flat_grn[changed_grn_indexes],min_val,max_val)
		else:
			None
	else:
		None
	grn = flat_grn.reshape(original_shape)
	return(grn)

##### ----- #####
def changeGRNLink(link_value,min_val,max_val):
		if link_value:
			new_value = 0
		else:
			range_size = max_val - min_val
			new_value = range_size * np.random.random() + min_val
		return(new_value)

##### ----- #####
changeGRNLink_vect = np.vectorize(changeGRNLink)

##### ----- #####
def mutateLink(link_value,link_mutation_bounds): # If a numpy array is passed, it's vectorized
	min_val,max_val = min(link_mutation_bounds),max(link_mutation_bounds)
	range_size = max_val-min_val
	result = link_value + range_size * np.random.random() + min_val
	return(result)

########## ----- SEQUENCE RELATED ----- ##########
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

##### ----- #####
def mutateBase(base):				# This mutation function is equivalent to
	Bases = ("T","C","A","G")		# the JC model of sequence evolution
	change = [x for x in Bases if x != base]
	new_base = np.random.choice(change)
	return(new_base)


#CHAPTER 5. THE DEVELOPMENT FUNCTION

##### ----- #####
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


# CHAPTER 6. FITNESS FUNCTIONS

##### ----- ##### (MAIN FUNC)
def calcFitness(development):
	is_alive = lastGeneExpressed(development,min_reproducin)
	genes_on = propGenesOn(development)
	exp_stab = expressionStability(development)
	sim_to_exp = exponentialSimilarity(development)
	fitness_val = is_alive * np.mean([genes_on,exp_stab,sim_to_exp])
	return(fitness_val)

##### ----- #####
# Is the last gene ever expressed at a certain level, and also expressed in the last step?
def lastGeneExpressed(development,min_reproducin):
	dev_steps, num_genes = development.shape
	last_col_bool = development[:,(num_genes - 1)] > min_reproducin
	last_val_last_col = development[dev_steps - 1, (num_genes - 1)]
	if last_col_bool.any() and last_val_last_col > 0:
		return_val = True
	else:
		return_val = False
	return(return_val)

##### ----- #####
# What proportion of the genes is on?
def propGenesOn(development):
	genes_on = development.sum(axis=0) > 0
	return(genes_on.mean())

##### ----- #####
# How stable is the expression throughout development?
def expressionStability(development):			# I haven't thought deeply about this.
	row_sums = development.sum(axis=1)		# What proportion of the data range is
	stab_val = row_sums.std() / (row_sums.max() - row_sums.min()) # the stdev? Less = better
	return(stab_val)

##### ----- #####
# How similar are the gene expression profiles to an exponential curve?
def exponentialSimilarity(development):
	dev_steps, num_genes = development.shape
	row_means = development.mean(axis=1)
	tot_dev_steps = dev_steps
	fitted_line = scipy.stats.linregress(range(tot_dev_steps),np.log(row_means))
	r_squared = fitted_line.rvalue ** 2
	return(r_squared)





###### FUNCTION SEMATARY #######
#class Offspring(object):
#	def __init__(self,name,parent):
#		self.name = name
#		self.parent = parent

#def mutateOrganism(old_organism):
#	organism = Organism(old_organism.name) # this is not a good idea. change.
#	organism.generation = organism.generation + 1
#	organism.grn = mutateGRN(organism.grn,grn_mutation_rate,link_mutation_bounds,prob_grn_change,new_link_bounds)
#	organism.decays = de_negativize(mutateGRN(organism.decays,decay_mutation_rate,thresh_decay_mut_bounds,0,None))
#	organism.thresholds = de_negativize(mutateGRN(organism.thresholds,thresh_mutation_rate,thresh_decay_mut_bounds,prob_thresh_change,thresh_decay_mut_bounds))
#	organism.development = develop(organism.start_vect,organism.grn,organism.decays,organism.thresholds,organism.dev_steps)
#	organism.fitness = fitness = calcFitness(organism.development)
#	if organism.fitness != 0:
#		organism.sequences = mutateGenome(organism.sequences,seq_mutation_rate)
#	else:
#		None
#	return(organism)


	# NOT SURE I WANT TO USE THIS YET...
#	def mutate(self):
#		self.generation = self.generation + 1
#		self.grn = mutateGRN(self.grn,grn_mutation_rate,link_mutation_bounds,prob_grn_change,new_link_bounds)
#		self.decays = de_negativize(mutateGRN(self.decays,decay_mutation_rate,thresh_decay_mut_bounds,0,None))
#		self.thresholds = de_negativize(mutateGRN(self.thresholds,thresh_mutation_rate,thresh_decay_mut_bounds,prob_thresh_change,thresh_decay_mut_bounds))
#		self.development = develop(self.start_vect,self.grn,self.decays,self.thresholds,self.dev_steps)
#		self.fitness = calcFitness(self.development)
#		if self.fitness != 0:
#			self.sequences = mutateGenome(self.sequences,seq_mutation_rate)
#		else:
#			None


