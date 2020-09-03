# PROLOGUE. IMPORTING STUFF

import numpy as np
import scipy
import random
import copy
import params_file as pf # Must add a check the the file exists, and that the
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
		self.individuals = producePop(self.pop_size,self.parent)
	def remove_dead(self):
		fitnesses = np.array([ x.fitness for x in self.individuals ])
		self.individuals = self.individuals[ fitnesses > 0 ]
		self.pop_size = self.individuals.size

# CHAPTER 2. MAIN FUNCTIONS TO CREATE AN ORGANISM, AND A POPULATION
# -- from scratch, or as a next generation

##### ----- #####
def makeNewOrganism(parent=None):
	if parent: 		# add also: if type(parent) is Organism:
		prob_grn_change = pf.prob_grn_change
		prob_thresh_change = pf.prob_thresh_change
		grn_mutation_rate = pf.grn_mutation_rate
		thresh_mutation_rate = pf.thresh_mutation_rate
		decay_mutation_rate = pf.decay_mutation_rate
		thresh_decay_mut_bounds = pf.thresh_decay_mut_bounds
		new_link_bounds = pf.new_link_bounds
		link_mutation_bounds = pf.link_mutation_bounds
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
			seq_mutation_rate = pf.seq_mutation_rate
			sequences = mutateGenome(parent.sequences,seq_mutation_rate)
		out_org = Organism(name,generation,parent.num_genes,parent.prop_unlinked,parent.prop_no_threshold,parent.thresh_boundaries,parent.decay_boundaries,parent.dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences)	
	else:
		num_genes = pf.num_genes
		decay_boundaries = pf.decay_boundaries
		prop_no_threshold = pf.prop_no_threshold
		thresh_boundaries = pf.thresh_boundaries
		prop_unlinked = pf.prop_unlinked
		dev_steps = pf.dev_steps
		name = "Lin" + str(int(np.random.random() * 1000000)) + "gen0"
		decays = randomMaskedVector(num_genes,0,decay_boundaries[0],decay_boundaries[1]) #BUG: check why some values are zero. Decays must never be zero
		thresholds = randomMaskedVector(num_genes,prop_no_threshold,thresh_boundaries[0],thresh_boundaries[1])
		start_vect = makeStartVect(num_genes)
		grn = makeGRN(num_genes,prop_unlinked)
		development = develop(start_vect,grn,decays,thresholds,dev_steps)
		fitness = calcFitness(development)
		if fitness == 0:
			sequences = None
		else:
			seq_length = pf.seq_length
			base_props = pf.base_props
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
def makeRandomSequence(seq_length,base_props=(0.25,0.25,0.25,0.25)):
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
def rand_bin_vect(total_vals,prop_zero):
	binVect = np.random.choice((0,1),total_vals,p=(prop_zero,1-prop_zero))
	return(binVect)

##### ----- #####
def randomMaskedVector(num_vals,prop_zero=0,min_val=0,max_val=1):
	if min_val > max_val:
		print("Error: minimum value greater than maximum value")
		return
	range_size = max_val - min_val
	if prop_zero == 0:
		rpv = np.array(range_size * np.random.random(num_vals) + min_val)
	else:
		mask = rand_bin_vect(num_vals,prop_zero)
		rpv = np.array(range_size * np.random.random(num_vals) + min_val)
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
		changer_vector = np.random.choice((0,1),flat_grn.size,p=(1-change_rate,change_rate))
		num_changes = sum(changer_vector)
		if num_changes == 0:
			None
		else:
			inactives = np.where(flat_grn == 0)[0]
#			prop_inactive = inactives.size/flat_grn.size
			actives = np.where(flat_grn != 0)[0]
#			prop_active = actives.size/flat_grn.size
			np.random.shuffle(actives)
			np.random.shuffle(inactives)
#			selector = np.random.choice((0,1),num_changes,p=(1-pf.prop_unlinked,pf.prop_unlinked))
#			selector = np.random.choice((0,1),num_changes,p=(prop_active,prop_inactive))
			selector = np.random.choice((0,1),num_changes)
			from_inactives = selector.size - selector.sum()
			from_actives = selector.sum()
			change_indexes = np.hstack([inactives[0:from_inactives],actives[0:from_actives]])
			changed_vals = np.ndarray((change_indexes.size),dtype=np.object)
			for i in range(change_indexes.size):
				changed_vals[i] = changeGRNLink(flat_grn[change_indexes[i]],min(change_bounds),max(change_bounds))
			flat_grn[change_indexes] = changed_vals
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
	genome_length = genome.size
	if genome_length > 10000000:
		print("Danger: bases assessed for mutation is too big")
	ones_to_mutate = np.random.choice((0,1),genome_length,p=(1-seq_mutation_rate,seq_mutation_rate))
	num_to_mut = ones_to_mutate.sum()
	if num_to_mut:
		original_dimensions = genome.shape
		flat_seq = genome.flatten()
		new_bases = np.ndarray((num_to_mut),dtype=np.object)
		mutated_nucs = np.where(ones_to_mutate == 1)[0]
		for i in range(mutated_nucs.size):
			new_bases[i] = mutateBase(flat_seq[mutated_nucs[i]])
		flat_seq[mutated_nucs] = new_bases
		final_seq = flat_seq.reshape(original_dimensions)
	else:
		final_seq = genome
	return(final_seq)

def mutateGenomeNew(genome,seq_mutation_rate):
	genome_length = genome.size
	if genome_length > 99999999:
		num_to_mut = np.int(seq_mutation_rate * genome_length) #Could eventually be given more stochasticity
		ones_to_mutate = np.array(random.sample(range(genome_length),num_to_mut))
	else:
		bin_vector = np.random.choice((0,1),genome_length,p=(1-seq_mutation_rate,seq_mutation_rate))
		ones_to_mutate = np.where(ones_to_mutate == 1)
		num_to_mut = bin_vector.sum()
	if num_to_mut:
		original_dimensions = genome.shape
		flat_seq = genome.flatten()
		new_bases = np.ndarray((num_to_mut),dtype=np.object)
		mutated_nucs = np.where(ones_to_mutate == 1)[0]
		for i in range(mutated_nucs.size):
			new_bases[i] = mutateBase(flat_seq[mutated_nucs[i]])
		flat_seq[mutated_nucs] = new_bases
		final_seq = flat_seq.reshape(original_dimensions)
	else:
		final_seq = genome
	return(final_seq)

##### ----- #####
def mutateBase(base):				# This mutation function is equivalent to
	bases = ("T","C","A","G")		# the JC model of sequence evolution
	change = [x for x in bases if x != base]# CAN THIS BE VECTORIZED? (must include prob of no change)
	new_base = np.random.choice(change)
	return(new_base)


#CHAPTER 5. THE DEVELOPMENT FUNCTION

##### ----- #####
def develop(start_vect,grn,decays,thresholds,dev_steps):
	geneExpressionProfile = np.array([start_vect])
	#Running the organism's development, and outputting the results
	#in an array called geneExpressionProfile
	invect = start_vect
	for i in range(dev_steps):
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
	min_reproducin = pf.min_reproducin
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
	dev_steps,num_genes = development.shape
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
	dev_steps,num_genes = development.shape
	row_means = development.mean(axis=1)
	tot_dev_steps = dev_steps
	fitted_line = scipy.stats.linregress(range(tot_dev_steps),np.log(row_means))
	r_squared = fitted_line.rvalue ** 2
	return(r_squared)


# CHAPTER 7: SELECTION FUNCTION

##### ----- #####
def select(parental_pop,prop_survivors,select_strategy = "random"):
	num_parents = parental_pop.individuals.flatten().size
	num_survivors = np.int(num_parents * prop_survivors)
#	num_survivors = sum(np.random.choice((0,1),num_parents #+1?#,p=(1-prop_survivors,prop_survivors))) #IDEAL
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
			founder_pop = Population(pf.pop_size,founder)
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
		founder_pop = Population(pf.pop_size,founder)
		founder_pop.populate()
	curr_pop = founder_pop
	fitnesses = np.array([ indiv.fitness for indiv in curr_pop.individuals ])
	death_count[0] = sum(fitnesses == 0)
	fitnesses_no_zeroes = np.array([ x for i,x in enumerate(fitnesses) if x > 0 ])
	living_fitness_mean[0] = np.mean(fitnesses_no_zeroes)
	living_fitness_sd[0] = np.std(fitnesses_no_zeroes)
	select_strategy = pf.select_strategy
	for i in range(num_generations):
		print("Generation",i,"is currently having a beautiful life...")
		survivor_pop = select(curr_pop,pf.prop_survivors,select_strategy)
		curr_pop = reproduce(survivor_pop,pf.pop_size,"equal")
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

def base_mutator(old_base,trans_prob_row):
	bases = ('T','C','A','G')
	new_base = np.random.choice(bases,p=trans_prob_row)
	return(new_base)

def base_mutator(some_array):
	bases = ('T','C','A','G')
	new_base = np.random.choice(bases,p=trans_prob_row)
	return(new_base)

def base_mutator(old_base,model="JC",mut_rate = 1.1e-8): # mut_rate default from table 1.2 on p.5 of Yang's book
	bases = ('T','C','A','G')
	if model == "JC":
		lambda_t = mut_rate/3
		subst_matrix = np.array(([1-3*lambda_t,lambda_t,lambda_t,lambda_t],[lambda_t,1-3*lambda_t,lambda_t,lambda_t],[lambda_t,lambda_t,1-3*lambda_t,lambda_t],[lambda_t,lambda_t,lambda_t,1-3*lambda_t]))
	change_rates = subst_matrix[bases.index(old_base),:]
	new_base = np.random.choice(('T','C','A','G'),p=change_rates)
	return(new_base)


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

#def mutateGenome_old(genome,seq_mutation_rate):
#	original_dimensions = genome.shape
#	flat_seq = genome.flatten()
#	genome_length = flat_seq.size
#	ones_to_mutate = np.random.choice((0,1),genome_length,p=(1-seq_mutation_rate,seq_mutation_rate))
#	if sum(ones_to_mutate):
#		mutated_nucs = [i for i,x in enumerate(ones_to_mutate) if x == 1 ]
#		for i in mutated_nucs:
#			flat_seq[i] = mutateBase(flat_seq[i])
#	final_seq = flat_seq.reshape(original_dimensions)
#	return(final_seq)

#def mutateGRN_2(grn,mutation_rate,mutation_bounds,change_rate,change_bounds): # Func also used for thresholds + decays
#	original_shape = grn.shape
#	flat_grn = grn.flatten()
#	actives = np.where(flat_grn != 0)[0]
#	mutator_vector = np.random.choice((False,True),actives.size,p=(1-mutation_rate,mutation_rate))
#	if sum(mutator_vector) == 0:
#		curr_flat_grn = flat_grn
#	else:
#		muts_indexes = actives[mutator_vector]
#		new_vals = np.ndarray((muts_indexes.size),dtype=np.object)
#		for i in range(new_vals.size):
#			new_vals[i] = mutateLink(flat_grn[muts_indexes[i]],mutation_bounds)
#		flat_grn[muts_indexes] = new_vals
#		curr_flat_grn = flat_grn
#	changer_vector = np.random.choice((False,True),curr_flat_grn.size,p=(1-change_rate,change_rate))
#	num_changes = sum(changer_vector)
#	if num_changes == 0:
#		final_flat_grn = curr_flat_grn
#	else:
#		inactives = np.where(curr_flat_grn == 0)[0]
#		np.random.shuffle(actives)
#		np.random.shuffle(inactives)
#		select_array = np.array([inactives,actives])
#		selector = np.random.choice((0,1),num_changes,p=(1-pf.prop_unlinked,pf.prop_unlinked))
#		from_inactives = selector.size - sum(selector)
#		from_actives = sum(selector)
#		change_indexes = np.hstack(np.array([inactives[0:from_inactives],actives[0:from_actives]]))
#		changed_vals = np.ndarray((change_indexes.size),dtype=np.object)
#		for i in range(change_indexes.size):
#			changed_vals[i] = changeGRNLink(curr_flat_grn[change_indexes[i]],min(change_bounds),max(change_bounds))
#		curr_flat_grn[change_indexes] = changed_vals
#		final_flat_grn = curr_flat_grn
#	final_grn = final_flat_grn.reshape(original_shape)
#	return(final_grn)
		
# FROM INITIAL MUTATE GRN - CHANGE PART DID NOT TAKE INTO ACCOUNT DIFFERENT SPARSENESS
#			changed_grn_indexes = np.array([i for i,x in enumerate(to_change) if x == 1])
#			min_val,max_val = change_bounds
#			if sum(to_change) > 1:
#				flat_grn[changed_grn_indexes] = changeGRNLink_vect(flat_grn[changed_grn_indexes],min_val,max_val)
#			else:
#				flat_grn[changed_grn_indexes] = changeGRNLink(flat_grn[changed_grn_indexes],min_val,max_val)


