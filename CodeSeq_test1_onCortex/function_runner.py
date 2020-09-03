#!/usr/bin/python3

import numpy as np
import nutria as nut

# Function takes in the development values of a single time point and produces 2 main parameters for building a NN: (1) # of neurons, (2) a mtrix of 1's and 0's to define who's connected to whom (one matrix per layer intermediate).
def nn_parameter_calculator(devline):
	num_nonzero_values = sum(devline > 0)
	
