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
	def __init__(self,name,generation,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences,proteome):
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
		self.proteome = proteome

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
		start_vect = parent.start_vect
		seq_mutation_rate = pf.seq_mutation_rate
		sequences = mutateGenome(parent.sequences,seq_mutation_rate)
		proteome = translate_genome(sequences)
		grn,decays,thresholds,development,fitness = master_mutator(parent,sequences,proteome)
		out_org = Organism(name,generation,parent.num_genes,parent.prop_unlinked,parent.prop_no_threshold,parent.thresh_boundaries,parent.decay_boundaries,parent.dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences,proteome)
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
			proteome = None
		else:
			seq_length = pf.seq_length
#			base_props = pf.base_props
			sequences = makeCodingSequenceArray(seq_length,num_genes)
			proteome = translate_genome(sequences)
		out_org = Organism(name,0,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences,proteome)
	return(out_org)

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
# Non-coding sequences are no longer in use
def makeRandomSequence(seq_length,base_props=(0.25,0.25,0.25,0.25)):
	bases = ("T","C","A","G")
	sequence = np.random.choice(bases,seq_length,p=base_props)
	return(sequence)

##### ----- ##### Non-coding sequences are no longer in use
def makeRandomSequenceArray(seq_length,base_props,num_genes):
	vect_length = seq_length * num_genes
	seq_vect = makeRandomSequence(vect_length,base_props)
	seq_arr = seq_vect.reshape(num_genes,seq_length)
	return(seq_arr)

def makeCodingSequence(seq_length):
	if seq_length % 3:
		print("Sequence length",seq_length,"is not a multiple of 3.")
		seq_length = seq_length - (seq_length % 3)
		print("Rounding to", seq_length)
	codon_list = list(coding_codons.keys())
	num_codons = np.int(seq_length/3)
	out_seq = np.array(list(''.join(np.random.choice(codon_list,num_codons))))
	return(out_seq)
	
def makeCodingSequenceArray(seq_length,num_genes):
	if seq_length % 3:
		print("Sequence length",seq_length,"is not a multiple of 3.")
		seq_length = seq_length - (seq_length % 3)
		print("Rounding to", seq_length)
	vect_length = seq_length * num_genes
	seq_vect = makeCodingSequence(vect_length)
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

########## ----- MASTER MUTATOR ----- ##########
def master_mutator(parent,offsp_genome,offsp_proteome):
	num_genes = offsp_genome.shape[0]
	syn_muts_table,nonsyn_muts_table = syn_nonsyn_muts(parent.sequences,offsp_genome,parent.proteome,offsp_proteome)
	new_ko_muts = new_kos(parent.proteome,offsp_proteome)
	total_num_muts = syn_muts_table.sum() + nonsyn_muts_table[nonsyn_muts_table != 99999999].sum() + new_ko_muts
	active_genes = np.bool_(parent.genes_on * (nonsyn_muts_table != 99999999)) # Makes KO mutations effective
	if total_num_muts == 0:
		out_grn = np.array(parent.grn)
		out_decays = np.array(parent.decays)
		out_thresholds = np.array(parent.thresholds)
	else:
#		print(total_num_muts,"mutations must be done")
#		non_neutral_grn_cells = active_genes.reshape(1,num_genes).T.dot(active_genes.reshape(1,num_genes))
#		neutral_grn_cells = np.invert(non_neutral_grn_cells)
		curr_grn = np.array(parent.grn)
		curr_decays = np.array(parent.decays)
		curr_thresholds = np.array(parent.thresholds)
		for gene_index in range(num_genes):
			gene_syn_muts = syn_muts_table[gene_index]
			gene_nonsyn_muts = nonsyn_muts_table[gene_index] if nonsyn_muts_table[gene_index] != 99999999 else 0
#			print("gene",gene_index," - syn muts:",gene_syn_muts,"- nonsyn muts:",gene_nonsyn_muts)
			if (gene_syn_muts > 0) | (gene_nonsyn_muts > 0):
#				print("Sent for mutation")
				curr_grn,curr_decays,curr_thresholds = gene_mutator(gene_index,gene_syn_muts,gene_nonsyn_muts,curr_grn,curr_decays,curr_thresholds,active_genes)
			else:
				None
#				print("Not sent for mutation")
		out_grn = curr_grn
		out_decays = curr_decays
		out_thresholds = curr_thresholds
	out_dev = develop(parent.start_vect,out_grn,out_decays,out_thresholds,parent.dev_steps,active_genes)
	out_fitness = calcFitness(out_dev)
	return(out_grn,out_decays,out_thresholds,out_dev,out_fitness)

def gene_mutator(gene_index,num_syn_muts,num_nonsyn_muts,in_grn,in_decays,in_thresholds,active_genes):
	num_genes = active_genes.size
	if num_nonsyn_muts > 0:
#		print("gene",gene_index,"has nonsyn muts. checking if changes will be made")
		changes = np.random.choice((0,1),num_nonsyn_muts,p=(1-pf.prob_grn_change,pf.prob_grn_change))
		if changes.sum() > 0:
#			print(changes.sum(),"changes will be made")
			num_nonsyn_muts = num_nonsyn_muts - changes.sum()
			changer = True
		else:
#			print("Changes won't be made")
			changer = False
	else:
		changer = False
	if active_genes[gene_index]:
#		print("gene",gene_index,"is active")
		if num_nonsyn_muts > 0:
#			print("organizing data for nonsyn mutations")
			total_genes_active = active_genes.sum()
			index_act_genes = np.hstack(np.where(active_genes))
			nonsyn_grn_sites_rows = np.hstack([np.repeat(gene_index,total_genes_active),index_act_genes[index_act_genes != gene_index]])
			nonsyn_grn_sites_cols = np.hstack([np.array(index_act_genes),np.repeat(gene_index,(nonsyn_grn_sites_rows.size-index_act_genes.size))])
			num_nonsyn_muts_options = nonsyn_grn_sites_rows.size + 2 # All nonsyn sites + decays + thresholds
			if num_nonsyn_muts_options < num_nonsyn_muts:
				to_mutate = np.random.choice(range(num_nonsyn_muts_options),num_nonsyn_muts,replace=True)
			else:
				to_mutate = np.random.choice(range(num_nonsyn_muts_options),num_nonsyn_muts,replace=False)
#			print("gene",gene_index,"nonsyn muts:",to_mutate)
			for i in to_mutate:
				if i < nonsyn_grn_sites_rows.size:
#					print("	",i,"nonsyn mut is in grn")
					row = nonsyn_grn_sites_rows[i]
					col = nonsyn_grn_sites_cols[i]
#					print("	nonsyn mut site: row -",row,", col -",col)
					in_grn[row][col] = mutateLink(in_grn[row][col],pf.link_mutation_bounds)
				elif i == nonsyn_grn_sites_rows.size:
#					print("	",i,"nonsyn mut is in decays")
					in_decays[gene_index] = de_negativize(mutateLink(in_decays[gene_index],pf.thresh_decay_mut_bounds))
				elif i == nonsyn_grn_sites_rows.size + 1:
#					print("	",i,"nonsyn mut is in thresholds")
					in_thresholds[gene_index] = de_negativize(mutateLink(in_thresholds[gene_index],pf.thresh_decay_mut_bounds))
		if num_syn_muts > 0:
#			print("organizing data for syn mutations")
			inactive_genes = np.invert(active_genes)
			total_genes_inactive = inactive_genes.sum()
			index_inact_genes = np.hstack(np.where(inactive_genes))
			syn_grn_sites_rows = np.hstack([np.repeat(gene_index,total_genes_inactive),index_inact_genes[index_inact_genes != gene_index]])
			syn_grn_sites_cols = np.hstack([np.array(index_inact_genes),np.repeat(gene_index,(syn_grn_sites_rows.size-index_inact_genes.size))])
			num_syn_muts_options = syn_grn_sites_rows.size + 2 # All syn sites + decays + thresholds
			if num_syn_muts_options == 0:
				to_mutate = None
			elif (num_syn_muts_options < num_syn_muts) & (num_syn_muts_options != 0):
				to_mutate = np.random.choice(range(num_syn_muts_options),num_syn_muts,replace=True)
			elif num_syn_muts_options >= num_syn_muts:
				to_mutate = np.random.choice(range(num_syn_muts_options),num_syn_muts,replace=False)
#			print("gene",gene_index,"syn muts:",to_mutate)
			if to_mutate is not None:
				for i in to_mutate:
					if i < syn_grn_sites_rows.size:
#						print("	",i,"syn mut is in grn")
						row = syn_grn_sites_rows[i]
						col = syn_grn_sites_cols[i]
#						print("	syn mut site: row -",row,", col -",col)
						in_grn[row][col] = mutateLink(in_grn[row][col],pf.link_mutation_bounds)
					elif i == syn_grn_sites_rows.size:
#						print("	",i,"syn mut is in decays")
						in_decays[gene_index] = de_negativize(mutateLink(in_decays[gene_index],pf.thresh_decay_mut_bounds))
					elif i == syn_grn_sites_rows.size + 1:
#						print("	",i,"syn mut is in thresholds")
						in_thresholds[gene_index] = de_negativize(mutateLink(in_thresholds[gene_index],pf.thresh_decay_mut_bounds))
	else:
#		print("gene",gene_index,"is NOT active, organizing all mutations as synonymous")
		num_syn_muts = num_syn_muts + num_nonsyn_muts
		num_nonsyn_muts = 0
		syn_grn_sites_rows = np.hstack([np.repeat(gene_index,num_genes),np.delete(np.array(range(num_genes)),gene_index,0)])
		syn_grn_sites_cols = np.hstack([np.array(range(num_genes)),np.repeat(gene_index,(num_genes-1))])
		num_syn_muts_options = syn_grn_sites_rows.size + 2 # All syn sites + decays + thresholds
		if num_syn_muts_options < num_syn_muts:
			to_mutate = np.random.choice(range(num_syn_muts_options),num_syn_muts,replace=True)
		else:
			to_mutate = np.random.choice(range(num_syn_muts_options),num_syn_muts,replace=False)
#		print("gene",gene_index,"inact syn muts:",to_mutate)
		for i in to_mutate:
			if i < syn_grn_sites_rows.size:
#				print("	",i,"inact syn mut is in grn")
				row = syn_grn_sites_rows[i]
				col = syn_grn_sites_cols[i]
#				print("	inact syn mut site: row -",row,", col -",col)
				in_grn[row][col] = mutateLink(in_grn[row][col],pf.link_mutation_bounds)
			elif i == syn_grn_sites_rows.size:
#				print("	",i,"inact syn mut is in decays")
				in_decays[gene_index] = de_negativize(mutateLink(in_decays[gene_index],pf.thresh_decay_mut_bounds))
			elif i == syn_grn_sites_rows.size + 1:
#				print("	",i,"inact syn mut is in thresholds")
				in_thresholds[gene_index] = de_negativize(mutateLink(in_thresholds[gene_index],pf.thresh_decay_mut_bounds))
	if changer:
#		print("doing changes")
		change_grn_sites_rows = np.hstack([np.repeat(gene_index,num_genes),np.delete(np.array(range(num_genes)),gene_index,0)])
		change_grn_sites_cols = np.hstack([np.array(range(num_genes)),np.repeat(gene_index,(num_genes-1))])
		num_change_options = change_grn_sites_rows.size
		change_index = np.random.choice(range(num_change_options),changes.sum(),replace=False)
#		print("Sites to change",change_index)
		for i in change_index:
			row = change_grn_sites_rows[i]
			col = change_grn_sites_cols[i]
#			print("change will happen on row:",row,"col:",col)
			min_val,max_val = pf.new_link_bounds
			in_grn[row][col] = changeGRNLink(in_grn[row][col],min_val,max_val)
	mod_grn = in_grn
	mod_decays = in_decays
	mod_thresholds = in_thresholds	
	return(mod_grn,mod_decays,mod_thresholds)
	

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
def translate_seq(sequence):
	start = 0
	stop = 3
	codons = np.int(sequence.size / 3)
	translated_seq = np.ndarray(codons,dtype='<U1')
	for i in range(codons):
		curr_cod = ''.join(sequence[start:stop])
		translated_seq[i] = gencode[curr_cod]
		start = start + 3
		stop = stop + 3
	return(translated_seq)

def translate_genome(in_genome):
	num_genes,num_bases = in_genome.shape[0],in_genome.shape[1]
	out_genome = np.ndarray((num_genes,(np.int(num_bases/3))),dtype='<U1')
	for i in range(num_genes):
		out_genome[i] = translate_seq(in_genome[i])
	return(out_genome)

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

def syn_nonsyn_muts(p_genome,o_genome,p_proteome,o_proteome):
	all_muts = np.where(p_genome != o_genome)
	nonsyns = np.where(p_proteome != o_proteome)
	num_genes = p_proteome.shape[0]
	syn_array = np.ndarray(num_genes,dtype=np.object)
	nonsyn_array = np.ndarray(num_genes,dtype=np.object)
	if all_muts[0].size != 0:
		total_mutated_genes = all_muts[0]
		nonsyn_mutated_genes = nonsyns[0]
		for i in range(num_genes):
			if np.any(o_proteome[i] == "_"):
				nonsyn_array[i] = 99999999
				# (BELOW) If gene has a KO mutation, all mutations are synonymous,
				# except if the KO is new. In this case the KO is subtracted from
				# the amount of synonymous muts.
				if np.any(p_proteome[i] == "_"):
					syn_array[i] = np.count_nonzero(total_mutated_genes == i)
				else:
					syn_array[i] = np.count_nonzero(total_mutated_genes == i) - 1
			else:
				nonsyn_array[i] = np.count_nonzero(nonsyn_mutated_genes == i)
				syn_array[i] = np.count_nonzero(total_mutated_genes == i) - nonsyn_array[i]
	else:
		syn_array[:] = 0
		nonsyn_array[:] = 0
	return(syn_array,nonsyn_array)

def new_kos(p_proteome,o_proteome):
	if np.any(o_proteome == "_"):
		outnum = 0
		for i in range(p_proteome.shape[0]):
			p_gene_ko = np.any(p_proteome[i] == "_")
			o_gene_ko = np.any(o_proteome[i] == "_")
			if (p_gene_ko == False) & (o_gene_ko == True):
				outnum += 1
	else:
		outnum = 0
	return(outnum)
	

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







##### ----- #####
#def makeNewOrganismOld(parent=None): # AN OUTDATED VERSION THAT I SCREWED UP LATER BY CONFUSING IT WITH THE REAL ONE
#	if parent: 		# add also: if type(parent) is Organism:
#		prob_grn_change = pf.prob_grn_change
#		prob_thresh_change = pf.prob_thresh_change
#		grn_mutation_rate = pf.grn_mutation_rate
#		thresh_mutation_rate = pf.thresh_mutation_rate
#		decay_mutation_rate = pf.decay_mutation_rate
#		thresh_decay_mut_bounds = pf.thresh_decay_mut_bounds
#		new_link_bounds = pf.new_link_bounds
#		link_mutation_bounds = pf.link_mutation_bounds
#		generation = parent.generation + 1
#		name = parent.name.split("gen")[0] + "gen" + str(generation)
#		start_vect = parent.start_vect
#		seq_mutation_rate = pf.seq_mutation_rate
#		sequences = mutateGenome(parent.sequences,seq_mutation_rate)
#		proteome = translate_genome(sequences)
#		grn,decays,thresholds,development,fitness = master_mutator(parent,sequences,proteome)
#		out_org = Organism(name,generation,parent.num_genes,parent.prop_unlinked,parent.prop_no_threshold,parent.thresh_boundaries,parent.decay_boundaries,parent.dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences,proteome)
#	else:
#		num_genes = pf.num_genes
#		decay_boundaries = pf.decay_boundaries
#		prop_no_threshold = pf.prop_no_threshold
#		thresh_boundaries = pf.thresh_boundaries
#		prop_unlinked = pf.prop_unlinked
#		dev_steps = pf.dev_steps
#		name = "Lin" + str(int(np.random.random() * 1000000)) + "gen0"
#		decays = randomMaskedVector(num_genes,0,decay_boundaries[0],decay_boundaries[1]) #BUG: check why some values are zero. Decays must never be zero
#		thresholds = randomMaskedVector(num_genes,prop_no_threshold,thresh_boundaries[0],thresh_boundaries[1])
#		start_vect = makeStartVect(num_genes)
#		grn = makeGRN(num_genes,prop_unlinked)
#		development = develop(start_vect,grn,decays,thresholds,dev_steps)
#		fitness = calcFitness(development)
#		if fitness == 0:
#			sequences = None
#			proteome = None
#		else:
#			seq_length = pf.seq_length
#			base_props = pf.base_props
#			sequences = makeCodingSequenceArray(seq_length,num_genes)
#			proteome = translate_genome(sequences)
#		out_org = Organism(name,0,num_genes,prop_unlinked,prop_no_threshold,thresh_boundaries,decay_boundaries,dev_steps,decays,thresholds,start_vect,grn,development,fitness,sequences,proteome)
#	return(out_org)

##### ----- #####


#def offspringNumTuple(tot_offspring,num_survivors,equal_fertility):
	# returns a tuple in which each element i is the amount of offspring the ith best fitting organism will have

#### IDEA ####
# When a new population is made, determine the population
# size from a random draw of a normal distribution with
# mean pop_size and stdev pop_stdev

###### FUNCTION SEMATARY #######

#def develop_old(start_vect,grn,decays,thresholds,dev_steps):
#	geneExpressionProfile = np.array([start_vect])
	#Running the organism's development, and outputting the results
	#in an array called geneExpressionProfile
#	invect = start_vect
#	for i in range(dev_steps):
#		decayed_invect = exponentialDecay(invect,decays)
#		currV = grn.dot(invect) - thresholds	      # Here the threshold is subtracted.
#		currV = de_negativize(currV) + decayed_invect # Think about how the thresholds
							      # should affect gene expression
#		geneExpressionProfile = np.append(geneExpressionProfile,[currV],axis=0)
#		invect = currV
#	return(geneExpressionProfile)


#def mutateGenomeNew(genome,seq_mutation_rate):
#	genome_length = genome.size
#	if genome_length > 99999999:
#		num_to_mut = np.int(seq_mutation_rate * genome_length) #Could eventually be given more stochasticity
#		ones_to_mutate = np.array(random.sample(range(genome_length),num_to_mut))
#	else:
#		bin_vector = np.random.choice((0,1),genome_length,p=(1-seq_mutation_rate,seq_mutation_rate))
#		ones_to_mutate = np.where(ones_to_mutate == 1)
#		num_to_mut = bin_vector.sum()
#	if num_to_mut:
#		original_dimensions = genome.shape
#		flat_seq = genome.flatten()
#		new_bases = np.ndarray((num_to_mut),dtype=np.object)
#		mutated_nucs = np.where(ones_to_mutate == 1)[0]
#		for i in range(mutated_nucs.size):
#			new_bases[i] = mutateBase(flat_seq[mutated_nucs[i]])
#		flat_seq[mutated_nucs] = new_bases
#		final_seq = flat_seq.reshape(original_dimensions)
#	else:
#		final_seq = genome
#	return(final_seq)



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


