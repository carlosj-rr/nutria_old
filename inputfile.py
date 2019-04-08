# CONSTRUCTION PARAMETERS
num_genes = 30
seq_length = 1000
prop_unlinked = 0.7
prop_no_threshold = 0.5
thresh_boundaries = (0.2,2)
decay_boundaries = (0,2)
dev_steps = 20 # For the moment, no more than 999 is possible
base_props = (0.25,0.25,0.25,0.25) # T,C,A,G
min_reproducin = 0.1
pop_size = 100
pop_stdev = 10

# MUTATION PARAMETERS
thresh_decay_mut_bounds = (-0.01,0.01)
thresh_mutation_rate = num_genes * (1-prop_no_threshold) / 100 # It can also be 0.001, for example
prob_thresh_change = 0.0001
decay_mutation_rate = num_genes / 100
seq_mutation_rate = 0.001	# Mutation likelihood per base, per generation.  Ex:
				# 1 mutation per 10,000 bases per generation: 1/10000 = 0.0001
				# mutation_ratios = () # For now, JC model hardcoded.

grn_mutation_rate = num_genes * (1-prop_unlinked) / 100	# Mutation likelihood per gene interaction, per generation.
				# Ex: 1 mutated interaction per 1,000 **interactions** 
				# --> 1/1000 = 0.001 (notice that non-interactions are not taken
				# into account, so it doesn't represent the proportion of the grn
				# that will mutate)
link_mutation_bounds = (-0.01,0.01)
prob_grn_change = 0.001 	# Probability that grn mutation will change grn structure
				# (i.e. it will create a new link or remove an existing one).
new_link_bounds = (-2,2)

# SELECTION PARAMETERS
prop_survivors = 0.1
tot_offspring = pop_size
