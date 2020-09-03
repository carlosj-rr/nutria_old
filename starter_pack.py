import nutria
import numpy as np
founder = nutria.makeNewOrganism()
while founder.fitness == 0:
	founder = nutria.makeNewOrganism()

baby1 = nutria.makeNewOrganism(founder)

