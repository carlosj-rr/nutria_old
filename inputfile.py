# CONSTRUCTION PARAMETERS
num_genes = 10
seq_length = 2000
prop_unlinked = 0.7
prop_no_threshold = 0.5
thresh_min,thresh_max = 0.2,2
decay_min,decay_max = 0,2
dev_steps = 15
base_props = (0.25,0.25,0.25,0.25) # T,C,A,G
min_reproducin = 0

# MUTATION PARAMETERS
thresh_decay_mut_bounds = (-0.01,0.01)
thresh_mutation_rate = 0.001
prob_thresh_change = 0.0001
decay_mutation_rate = 0.001
seq_mutation_rate = 0.001	# Mutation likelihood per base, per generation.  Ex:
				# 1 mutation per 10,000 bases per generation: 1/10000 = 0.0001
				# mutation_ratios = () # For now, JC model hardcoded.

grn_mutation_rate = 0.001 	# Mutation likelihood per gene interaction, per generation.
				# Ex: 1 mutated interaction per 1,000 **interactions** 
				# --> 1/1000 = 0.001 (notice that non-interactions are not taken
				# into account, so it doesn't represent the proportion of the grn
				# that will mutate)
link_mutation_bounds = (-0.01,0.01)
prob_grn_change = 0.0001 	# Probability that grn mutation will change grn structure
				# (i.e. it will create a new link or remove an existing one).
new_link_bounds = (-2,2)
