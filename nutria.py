# PROLOGUE. IMPORTING STUFF

import numpy as np
import scipy
import random
import params_file as pf # Must add a check the the file exists, and that the
			 # variables are valid
from numpy import exp
from scipy import stats
import math


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
		self.genes_on = np.int_(self.development.sum(axis=0) != 0)
		self.fitness = fitness
		self.sequences = sequences
		self.num_mutable_values = (self.num_genes*2)+1

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

# CHAPTER 1a: Some other declarations:
gencode = {
'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

coding_codons = {
'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
'TAC':'Y', 'TAT':'Y',
'TGC':'C', 'TGT':'C', 'TGG':'W',
}

# CHAPTER 2. MAIN FUNCTIONS TO CREATE AN ORGANISM, AND A POPULATION
# -- from scratch, or as a next generation

##### ----- ##### Produces random founders until one has a fitness value of more than min_fitness_val
def founderFinder(min_fitness_val=0):
	founder = makeNewOrganism()
	while founder.fitness <= min_fitness_val:
		founder = makeNewOrganism()
	return(founder)

##### ----- ##### the core function to create an organism, can be from scratch, or as a mutated version of a previously existing organism.
def makeNewOrganism(parent=None):
	if parent: 		# add also: if type(parent) is Organism:
		thresh_mutation_rate = pf.thresh_mutation_rate
		decay_mutation_rate = pf.decay_mutation_rate
		thresh_decay_mut_bounds = pf.thresh_decay_mut_bounds
		new_link_bounds = pf.new_link_bounds
		link_mutation_bounds = pf.link_mutation_bounds
		generation = parent.generation + 1
		name = parent.name.split("gen")[0] + "gen" + str(generation)
		start_vect = parent.start_vect
		seq_mutation_rate = pf.seq_mutation_rate
		sequences = mutateGenome(parent.sequences,seq_mutation_rate)
		grn,decays,thresholds,development,fitness = master_mutator(np.array(parent.sequences),np.array(parent.start_vect),np.int(parent.dev_steps),np.int(parent.num_mutable_values),np.array(parent.grn),np.array(parent.decays),np.array(parent.thresholds),sequences)
		out_org = Organism(name,generation,parent.num_genes,parent.prop_unlinked,parent.prop_no_threshold,parent.thresh_boundaries,parent.decay_boundaries,parent.dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences)
	else:
		num_genes = pf.num_genes
		decay_boundaries = pf.decay_boundaries
		prop_no_threshold = pf.prop_no_threshold
		thresh_boundaries = pf.thresh_boundaries
		prop_unlinked = pf.prop_unlinked
		dev_steps = pf.dev_steps
		name = "Lin" + str(int(np.random.random() * 1000000)) + "gen0"
		decays = randomMaskedVector(num_genes,0,decay_boundaries[0],decay_boundaries[1]) #BUG: check why some values are zero. Decays must never be zero ::--:: not sure this is still happening (7/9/2020 - CJR)
		thresholds = randomMaskedVector(num_genes,prop_no_threshold,thresh_boundaries[0],thresh_boundaries[1])
		start_vect = makeStartVect(num_genes)
		grn = makeGRN(num_genes,prop_unlinked)
		development = develop(start_vect,grn,decays,thresholds,dev_steps)
		fitness = calcFitness(development)
		if fitness == 0:
			sequences = None
		else:
			param_number=(pf.num_genes*2)+1
			if pf.seq_length % param_number:
				print("Sequence length",pf.seq_length,"is not a multiple of parameter number",param_number,", Changing to a better number...")
				seq_length = pf.seq_length - (pf.seq_length % param_number)
				print("sequence length changed to",seq_length)
				pf.seq_length=seq_length
			else:
				seq_length = pf.seq_length
			sequences = makeRandomSequenceArray(pf.seq_length,pf.base_props,num_genes)
		out_org = Organism(name,0,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences)
	return(out_org)

#ACHTUNG: Produces an np.array for the Population() class constructor. All original populations must be made with the class constructor, otherwise it will not behave like a Population() object. Remember, Population objects have to be 'populated' (class function .populate).
def producePop(pop_size,parent=None):
	popu = np.ndarray((pop_size,),dtype=np.object)
	if not parent:
		for i in range(pop_size):
			popu[i] = makeNewOrganism()
	else:
		if type(parent) is Organism:
			for i in range(pop_size):
#				print("Making member number",i,"of the new population")
				popu[i] = makeNewOrganism(parent)
		else:
			print("The type of the parent is not correct",type(parent))
	return(popu)


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

##### ----- #####

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

########## ----- MASTER MUTATOR ----- ########## (needs info from the parental organism, and its mutated genome)
def master_mutator(parental_sequences,start_vector,dev_steps,num_mutable_values,curr_grn,curr_decays,curr_thresholds,offsp_genome):
	num_mutable_params = num_mutable_values
	num_genes = pf.num_genes
	all_mutated_sites=np.array(np.where(parental_sequences != offsp_genome))
	genome_map = ranged_dictionary_maker(num_genes,num_mutable_values)
	mutated_sectors_list = mutated_sectors_mapper(genome_map,all_mutated_sites)
	total_num_muts = len(mutated_sectors_list)
	if total_num_muts == 0:
		out_grn = curr_grn
		out_decays = curr_decays
		out_thresholds = curr_thresholds
	else:
		genes_to_be_mutated = np.unique(all_mutated_sites[0])
		out_grn,out_decays,out_thresholds=GRN_sectorial_mutator(curr_grn,curr_decays,curr_thresholds,mutated_sectors_list,all_mutated_sites)
	out_dev = develop(start_vector,out_grn,out_decays,out_thresholds,dev_steps)
	out_fitness = calcFitness(out_dev)
	return(out_grn,out_decays,out_thresholds,out_dev,out_fitness)

#Produce a dictionary of addresses in order to locate single mutations to specific values. Independent for each gene (gene-gene interaction addresses must be adapted to the current gene being mutated). Function now produces a location map for the entire genome (7/9/2020 - CJRR).
def ranged_dictionary_maker(num_genes,num_mutable_vals):
	seq_length=pf.seq_length
	if seq_length % num_mutable_vals:
		print("Sequence length",seq_length,"is not a multiple of parameter length",num_mutable_vals)
		seq_length = seq_length - (seq_length % num_mutable_vals)
		pf.seq_length = seq_length
		print("Changing sequence length to",seq_length)
		block_size = np.int(seq_length/num_mutable_vals)
		print("This implies a block size of",block_size,"nt for each parameter")
	else:
		block_size = np.int(seq_length/num_mutable_vals)
	tot_dict_list=[]
	for w in range(num_genes):
		gene_dict_val_list=[('decay',w),('threshold',w)]
		addendum1=list(zip(range(num_genes),list(np.repeat(w,num_genes))))
		addendum2=list(zip(list(np.repeat(w,num_genes)),range(num_genes)))
		addendum2.remove((w,w))
		gene_dict_val_list=gene_dict_val_list+addendum1+addendum2
		block_size = np.int(seq_length/num_mutable_vals)
		a,b,rep = 0,block_size,0
		out_dict = {}
		while b <= seq_length:
			out_dict[range(a,b)] = gene_dict_val_list[rep]
			a=a+block_size
			b=b+block_size
			rep+=1
		tot_dict_list.append(out_dict)
	return(tot_dict_list)


def mutated_sectors_mapper(genome_map,all_mutated_sites):
	outlistoflists=[]
	genes_to_be_mutated=np.unique(all_mutated_sites[0])
	for i in genes_to_be_mutated:
		gene_index=i
		sites_tomut=all_mutated_sites[1][all_mutated_sites[0] == gene_index]
		outlist=[]
		ranged_dict=genome_map[gene_index]
		for val in sites_tomut:
			for key in ranged_dict:
				if val in key:
					outlist.append(ranged_dict[key])
		outlistoflists.append(outlist)
	outlistoflists=[item for elem in outlistoflists for item in elem]
	return(outlistoflists)


def GRN_sectorial_mutator(out_grn,out_decays,out_thresholds,mutated_sectors_list,all_mutated_sites):
	i=0
	for i in range(len(all_mutated_sites[0])-1):
		gene_index=all_mutated_sites[0][i]
		addresses_toMut=mutated_sectors_list[i]
		if 'decay' in addresses_toMut:
			out_decays[gene_index] = mutateLink(out_decays[gene_index],pf.thresh_decay_mut_bounds)
		elif 'threshold' in addresses_toMut:
			out_thresholds[gene_index] = mutateLink(out_thresholds[gene_index],pf.thresh_decay_mut_bounds)
		else:
			change_address=addresses_toMut
			out_grn[change_address]= mutateLink(out_grn[change_address],pf.link_mutation_bounds)
	return(out_grn,out_decays,out_thresholds)


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
		print("Danger: number of bases assessed for mutation is too large")
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
#		different_sites = np.where(genome != final_seq)
	else:
		final_seq = genome
#		different_sites = None
	return(final_seq)


##### ----- #####
def mutateBase(base):				# This mutation function is equivalent to
	bases = ("T","C","A","G")		# the JC model of sequence evolution
	change = [x for x in bases if x != base]# CAN THIS BE VECTORIZED? (must include prob of no change)
	new_base = np.random.choice(change)
	return(new_base)


#CHAPTER 5. THE DEVELOPMENT FUNCTION

##### ----- #####
def develop(start_vect,grn,decays,thresholds,dev_steps,nonsenses = 1):
	start_vect = start_vect * nonsenses
	geneExpressionProfile = np.array([start_vect])
	#Running the organism's development, and outputting the results
	#in an array called geneExpressionProfile
	invect = start_vect
	for i in range(dev_steps):
		decayed_invect = exponentialDecay(invect,decays)
		currV = grn.dot(invect) - thresholds	      # Here the threshold is subtracted.
		currV = de_negativize(currV) + decayed_invect # Think about how the thresholds
							      # should affect gene expression
		currV = currV * nonsenses
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
# How similar are the gene expression profiles to an exponential curve? (inverse points - better to be less exponential than more)
def exponentialSimilarity(development):
	dev_steps,num_genes = development.shape
	row_means = development.mean(axis=1)
	tot_dev_steps = dev_steps
	fitted_line = scipy.stats.linregress(range(tot_dev_steps),np.log(row_means))
	r_squared = fitted_line.rvalue ** 2
	return(r_squared)


# CHAPTER 7: SELECTION FUNCTION

##### ----- #####
def select(parental_pop,prop_survivors,select_strategy = "greedy"):
	num_parents = parental_pop.individuals.flatten().size
	num_survivors = np.int(num_parents * prop_survivors)
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
		red_flag=False #Intended to raise if the population is becoming endangered
	else:
		print("Watch out: Only",living_select_table[:,1].size,"offspring alive, but you asked for",num_survivors,"population is endangered")
		red_flag=True #Raise the red flag if the population is becoming endangered (i.e., user is asking for more selected fit individuals than there actually remain alive). Still unsure what to do with this, though.
		surviving_orgs = parental_pop.individuals[living_select_table[:,0].astype(int)]
	survivors_pop = Population(surviving_orgs.size)
	survivors_pop.individuals = surviving_orgs
	return(survivors_pop,red_flag)

def reproduce(survivors_pop,final_pop_size,reproductive_strategy="none"):
	survivors = survivors_pop.individuals
	offspring_per_parent = round(final_pop_size/survivors.size)
	final_pop_array = np.ndarray((survivors.size,offspring_per_parent),dtype=np.object)
	for i in range(survivors.size):
		for j in range(offspring_per_parent):
			final_pop_array[i][j] = makeNewOrganism(survivors[i])
	final_pop_array = final_pop_array.flatten()
	new_pop_indivs = final_pop_array.flatten()
	new_gen_pop = Population(new_pop_indivs.size)
	new_gen_pop.individuals = new_pop_indivs
	return(new_gen_pop)

# For the moment, not used. When vectorized, it can be used in an array of individuals
#def replicatorMutator(parent,num_offspring):
#	out_array = np.ndarray((num_offspring,),dtype=np.object)
#	for i in range(num_offspring):
#		out_array[i] = makeNewOrganism(parent)
#	return(out_array)
		
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
			founder_pop.remove_dead()
		elif type(founder) == Population:
			print("A founder population was provided")
			founder_pop = founder
			founder_pop.remove_dead()
		else:
			print("Error: A founder was provided but it is neither type Organism nor Population")
			return
	else:
		print("No founder provided, making founder Organism and Population")
		founder = founderFinder()
		founder_pop = Population(pf.pop_size,founder)
		founder_pop.populate()
		founder_pop.remove_dead()
	curr_pop = founder_pop
	fitnesses = np.array([ indiv.fitness for indiv in curr_pop.individuals ])
	death_count[0] = sum(fitnesses == 0)
	fitnesses_no_zeroes = np.array([ x for i,x in enumerate(fitnesses) if x > 0 ])
	living_fitness_mean[0] = np.mean(fitnesses_no_zeroes)
	living_fitness_sd[0] = np.std(fitnesses_no_zeroes)
	select_strategy = pf.select_strategy
	for i in range(num_generations):
		print("Generation",i,"is currently having a beautiful life...")
		survivor_pop,red_flag = select(curr_pop,pf.prop_survivors,select_strategy)
		curr_pop = reproduce(survivor_pop,pf.pop_size,"equal") # Must correct function for cases in which the survivors are less than the number of survivors defined by user (say, user wants 25 survivors, but only 10 remain, they need to reproduce differently to produce a new population of size 100) - DONE-ish (6/8/2020), now it gets a red flag, but stil haven't defined what happens next.
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

# The function that uses sine waves to determine a 'connectome' -- takes in two values (which will determine the frequencies of two sine waves), and it returns a 1D vector of 1's and 0's that can be reshaped into a 2D matrix that determines which neurons are connected with which.
def sin_01_vect(param1,param2,num_values):
	if param1 == 0 or param2 == 0:
		return([0] * num_values)
	else:
		omega_closest_to_0 = min(abs(param1),abs(param2))
		set_period = 2*math.pi/omega_closest_to_0
		step_size = set_period/num_values
		read_points = np.arange(0+(step_size/2),set_period,step_size)
		added_func_reads = np.sin(param1*read_points) + np.sin(param2*read_points)
		output = np.int_(added_func_reads > 0)
	return(output)

#*************** END OF THE PROGRAM *********************#









############### RECOMBINATION FUNCTIONS - TO WORK WITH LATER ###############


#def recombine_pop(individuals_array,recomb_pairing="panmictic"):	# Function INcomplete
#	if recomb_pairing == "panmictic":
#		#Recomb
#		None
#	else:
#		print("Recombination style",recomb_style,"not recognized")

#def recombine_pair(indiv_1,indiv_2,recomb_style="vertical"):		# Function complete
#	chiasma = np.random.choice(range(1,indiv_1.num_genes))
#	indiv_out = makeNewOrganism(indiv_1)
#	if recomb_style == "vertical":
#		indiv_out.grn = np.append(indiv_1.grn[:,:chiasma],indiv_2.grn[:,chiasma:],axis=1)
#	elif recomb_style == "horizontal":
#		indiv_out.grn = np.append(indiv_1.grn[:chiasma,:],indiv_2.grn[chiasma:,:],axis=0)
#	elif recomb_style == "minimal":
#		print("Minimal style of recombination still not programmed")
#	elif recomb_style == "maximal":
#		print("Maximal style of recombination still not programmed")
#	indiv_out.sequences = np.append(indiv_1.sequences[:chiasma],indiv_2.sequences[chiasma:],axis=0)
#	indiv_out.decays = np.append(indiv_1.decays[:chiasma],indiv_2.decays[chiasma:])
#	indiv_out.thresholds = np.append(indiv_1.thresholds[:chiasma],indiv_2.thresholds[chiasma:])
#	indiv_out.development = develop(indiv_out.start_vect,indiv_out.grn,indiv_out.decays,indiv_out.thresholds,indiv_out.dev_steps)
#	indiv_out.fitness = calcFitness(indiv_out.development)
#	return(indiv_out)

#def make_recomb_index_pairs(total_individuals):				# Function complete
#	first_col = np.array(range(total_individuals))
#	second_col = np.ndarray(total_individuals,dtype=np.object)
#	value_pool = list(range(total_individuals))
#	counter = 0
#	for i in first_col:
#		pair = np.random.choice([ x for x in value_pool if x != i ])
#		second_col[counter] = pair
#		value_pool = [ x for x in value_pool if x != pair ]
#		counter += 1
#	out_table = np.append(first_col,second_col).reshape(2,total_individuals).T
#	return(out_table)
	

############################################################################
