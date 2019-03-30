import numpy as np
from inputfile import * # Must add a check the the file exists, and that the
			# variables are valid
from numpy import exp

class NewOrganism(object):
	#Constructor
	def __init__(self,name, number_of_genes,propUnlinks,propNoThresholds,thresh_min,thresh_max,decay_min,decay_max,dev_steps):
		self.name = name
		self.number_of_genes = number_of_genes
		self.propUnlinks = propUnlinks
		self.propNoThresholds = propNoThresholds
		self.thresh_min = thresh_min
		self.thresh_max = thresh_max
		self.decay_min = decay_min
		self.decay_max = decay_max
		self.dev_steps = dev_steps
		self.decays = decays = randomMaskedVector(number_of_genes,0,decay_min,decay_max)
		self.thresholds = thresholds = randomMaskedVector(number_of_genes,propNoThresholds,thresh_min,thresh_max)
		start_vect = makeStartVect(number_of_genes)
		self.grn = grn = makeGRN(number_of_genes,propUnlinks)
		self.development = develop(start_vect,grn,decays,thresholds,dev_steps)
		self.sequences = sequences = makeRandomSequenceArray(seq_length,base_props,number_of_genes)


""" The population class below is of unrelated GRNs. This is only useful
when trying to find a founder. TO DO: add a way to construct population
based on a parent.

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
		pop[i] = NewOrganism(name,num_genes,prop_unlinked,prop_no_threshold,thresh_min,thresh_max,decay_min,decay_max,dev_steps)
	return(pop)

def makeGRN(numGenes,propUnlinks):
	grn = randomMaskedVector(numGenes ** 2,propUnlinks,-2,2)
	grn = grn.reshape(numGenes,numGenes)
	return(grn)

def makeRandomSequence(seq_length,base_props):
	bases = ("T","C","A","G")
	sequence = np.array(list(''.join(np.random.choice(bases,seq_length,p=base_props))))
	return(sequence)

def makeRandomSequenceArray(seq_length,base_props,num_genes):
	vect_length = seq_length * num_genes
	seq_vect = makeRandomSequence(vect_length,base_props)
	seq_arr = seq_vect.reshape(num_genes,seq_length)
	return(seq_arr)

def makeStartVect(numGenes):
	startingVect = np.array([1] * 1 + [0] * (numGenes - 1))
	return(startingVect)

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
	if minVal >= maxVal:
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


