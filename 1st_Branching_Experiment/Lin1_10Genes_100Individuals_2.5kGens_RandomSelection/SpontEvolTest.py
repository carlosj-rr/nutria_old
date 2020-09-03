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
		self.parent = parent
		self.individuals = None
	def populate(self):
		self.individuals = producePop(pop_size,self.parent)


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


# CHAPTER 3. SUPPORT FUNCTIONS FOR MAKING AND ORGANISM FROM SCRATCH

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
	if is_alive:
		genes_on = propGenesOn(development)
		exp_stab = expressionStability(development)
		sim_to_exp = exponentialSimilarity(development)
		fitness_val = np.mean([genes_on,exp_stab,sim_to_exp])
	else:
		fitness_val = 0
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


# CHAPTER 7: SELECTION FUNCTION

##### ----- #####
def select(parental_pop,prop_survivors,select_strategy = "random"):
	num_parents = parental_pop.individuals.flatten().size
	num_survivors = round(num_parents * prop_survivors)
	fitness_vals = np.array([ x.fitness for x in parental_pop.individuals ])
	select_table = np.array((np.array(range(fitness_vals.size)),fitness_vals)).T
	living_select_table = select_table[select_table[:,1] > 0]
	living_select_table = living_select_table[np.argsort(living_select_table[:,1])]
	num_parentals_alive = living_select_table[:,1].size
	if num_survivors <= num_parentals_alive:
		if select_strategy == "greedy":
			x = np.array(range(num_survivors)) + 1
			goods_table = living_select_table[num_parentals_alive - x,:] # ACHTUNG: This table is ordered decreasing!!
			surviving_orgs = parental_pop.individuals[goods_table[:,0].astype(int)]
		elif select_strategy == "random":
			randomers = np.random.random_integers(0,living_select_table[:,0].size-1,num_survivors)
			rand_indexes = living_select_table[randomers,0].astype(int)
			surviving_orgs = parental_pop.individuals[rand_indexes]
		else:
			print("Error: No selective strategy was provided")
			return
	else:
		print("Watch out: Only",living_select_table[:,1].size,"offspring alive, but you asked for",num_survivors)
		surviving_orgs = parental_pop.individuals[living_select_table[:,0].astype(int)]
	survivors_pop = Population(surviving_orgs.size)
	survivors_pop.individuals = surviving_orgs
	return(survivors_pop)

def reproduce(survivors_pop,final_pop_size,reproductive_strategy="equal"):
	survivors = survivors_pop.individuals
	if reproductive_strategy == "equal":
		offspring_per_parent = round(final_pop_size/survivors.size)
		final_pop_array = np.ndarray((survivors.size,offspring_per_parent),dtype=np.object)
		for i in range(survivors.size):
			for j in range(offspring_per_parent):
				final_pop_array[i][j] = makeNewOrganism(survivors[i])
		final_pop_array = final_pop_array.flatten()
	else:
		None
	new_pop_indivs = final_pop_array.flatten()
	new_gen_pop = Population(new_pop_indivs.size)
	new_gen_pop.individuals = new_pop_indivs
	return(new_gen_pop)

# For the moment, not used. When vectorized, it can be used in an array of individuals
def replicatorMutator(parent,num_offspring):
	out_array = np.ndarray((num_offspring,),dtype=np.object)
	for i in range(num_offspring):
		out_array[i] = makeNewOrganism(parent)
	return(out_array)
		
def runThisStuff(num_generations = 1000,founder=None):
	death_count = np.ndarray((num_generations + 1,),dtype=np.object)
	living_fitness_mean = np.ndarray((num_generations + 1,),dtype=np.object)
	living_fitness_sd = np.ndarray((num_generations + 1,),dtype=np.object)
	if founder:
		if type(founder) == Organism:
			print("A founder organism was provided")
			founder = founder
			founder_pop = Population(pop_size,founder)
			founder_pop.populate()	
		elif type(founder) == Population:
			print("A founder population was provided")
			founder_pop = founder
		else:
			print("Error: A founder was provided but it is neither type Organism nor Population")
			return
	else:
		print("No founder provided, making founder Organism and Population")
		founder = makeNewOrganism()
		while founder.fitness == 0:
			founder = makeNewOrganism()
		founder_pop = Population(pop_size,founder)
		founder_pop.populate()
	curr_pop = founder_pop
	fitnesses = np.array([ indiv.fitness for indiv in curr_pop.individuals ])
	death_count[0] = sum(fitnesses == 0)
	fitnesses_no_zeroes = np.array([ x for i,x in enumerate(fitnesses) if x > 0 ])
	living_fitness_mean[0] = np.mean(fitnesses_no_zeroes)
	living_fitness_sd[0] = np.std(fitnesses_no_zeroes)
	for i in range(num_generations):
		print("Generation",i,"is currently having a beautiful life...")
		survivor_pop = select(curr_pop,0.25,select_strategy)
		curr_pop = reproduce(survivor_pop,100,"equal")
		fitnesses = np.array([ indiv.fitness for indiv in curr_pop.individuals ])
		death_count[i + 1] = sum(fitnesses == 0)
		if death_count[i + 1]:
			fitnesses_no_zeroes = np.array([ x for i,x in enumerate(fitnesses) if x > 0 ])
		else:
			fitnesses_no_zeroes = fitnesses
		living_fitness_mean[i +  1] = np.mean(fitnesses_no_zeroes)
		living_fitness_sd[i + 1] = np.std(fitnesses_no_zeroes)
		print("Dead:",death_count[i + 1],"Fitness mean:",living_fitness_mean[i + 1],"Fitness_sd:",living_fitness_sd[i + 1])
	summary_table = np.array((death_count,living_fitness_mean,living_fitness_sd))
	return(summary_table.T,founder_pop,curr_pop)

##### EXPORTATION FUNCTIONS #####
def exportOrgSequences(organism,outfilename="outfile.fas"):
	with open(outfilename,"w") as outfile:
		for i in range(organism.num_genes):
			counter = i + 1
			gene_name = ">" + organism.name + "_gene" + str(counter)
			sequence = ''.join(organism.sequences[i])
			print(gene_name, file=outfile)
			print(sequence, file=outfile)

def exportAlignments(organism_array,outfile_prefix="outfile"):
	num_orgs = organism_array.size
	num_genes = np.max(np.array([ x.num_genes for x in organism_array ]))
	sequences_array = np.array([ x.sequences for x in organism_array ])
	for i in range(num_genes):
		filename = outfile_prefix + "_gene" + str(i+1) + ".fas"
		with open(filename,"w") as gene_file:
			for j in range(num_orgs):
				seq_name = ">" + organism_array[j].name + "_gene" + str(i+1) + "_" + str(j+1)
				sequence = ''.join(sequences_array[j,i,:])
				print(seq_name, file=gene_file)
				print(sequence, file=gene_file)
		print("Gene",i+1,"done")

##### TEST TO RUN: THE SAME AS I DID YESTERDAY (10 GENES, 15 STEPS, 5K GENERATION), BUT WITHOUT SELECTION.
##### TO SEE IF, LIKE IN YESTERDAY'S RUN, THE FINAL BEST GRN HAS MORE CONNECTIONS THAN THE FOUNDER
##### TO SEE IF IT'S THAT SUCH GRNs ARE FAVORED, OR NOT.

#def offspringNumTuple(tot_offspring,num_survivors,equal_fertility):
	# returns a tuple in which each element i is the amount of offspring the ith best fitting organism will have

#### IDEA ####
# When a new population is made, determine the population
# size from a random draw of a normal distribution with
# mean pop_size and stdev pop_stdev

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


