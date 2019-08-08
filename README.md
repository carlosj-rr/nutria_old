# nutria
An Artificial Life Simulator for Evo-Devo studies

# General Description
nutria is a prototype simulator of biological life. It simultaneously simulates development, gene regulation, and genomic encoding on each organism, and these elements are allowed to evolve through a classical genetic algorithm.

## A synthetic organism
Each organism is defined by a network of genes that have regulatory interactions with each other and themselves (a gene regulatory network, GRN), and each node in the GRN (= gene) has a protein-coding nucleotide sequence associated to it.

Concretely, this is an NxN matrix (for N genes), and each column represents the regulatory effects of a gene's expression on all other genes. This number can be zero (no effect), positive (upregulation), or negative (downregulation). The sequences are just an array of N coding sequences.

### Development
Development is simulated by multiplying (dot product) the GRN matrix by an arbitrary starting vector of size N (the starting amounts of each gene), and then repeating the operation with the output of this dot product for a user-defined amount of iterations. There are some details to this operation that I'm not explaining and which are already in the code but I will explain those in detail later, in the wiki page. The data resulting from this operation is a progression of N-size vectors that show the amounts of each gene as development progresses. This can be converted into a more specific phenotype, but for the moment, the 'gene expression profiles' are the phenotype.

### Mutation
The sequences of each gene are mutated under user-defined parameters, and are reflected in the GRN matrix. For example, if a mutation on Gene 1 occurs, and it is a synonymous mutation (i.e. it does not change the amino acid), a synonymous change is also done on any of the regulatory interactions of Gene 1 in the GRN matrix. The converse is true for non-synonymous mutations.

## Evolution
Populations of these organisms are created, and they are evolved for a used-defined amount of generations, and following a user-defined phylogeny. There is no recombination so far (although it is totally possible, since all genes are homologous...future prospect).

### Fitness
Fitness is defined with the 'phenotype' (= gene expression profiles). For the moment, it's defined by things like: the system doesn't go into exponential gene expression, gene expression is relatively stable, and one (arbitrarily chosen) gene is expressed at a given amount at the end of the development. This gene is an indicator of reproductive maturity.

### Selection
Selection for the moment is either random or highly stringent (only the top scoring X% survive).
