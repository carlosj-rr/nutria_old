>>> founder.genes_on
array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])

cst1,csf1,csa1 = nutria.runThisStuff(10000,founder); cst2,csf2,csa2 = nutria.runThisStuff(10000,founder); cst3,csf3,csa3 = nutria.runThisStuff(10000,csa1); cst4,csf4,csa4 = nutria.runThisStuff(10000,csa1); cst5,csf5,csa5 = nutria.runThisStuff(10000,csa2); cst6,csf6,csa6 = nutria.runThisStuff(10000,csa2)

>>> csa3.individuals[0].genes_on
array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
>>> csa4.individuals[0].genes_on
array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
>>> csa5.individuals[0].genes_on
array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
>>> csa6.individuals[0].genes_on
array([1, 1, 0, 1, 1, 0, 0, 1, 1, 1])

