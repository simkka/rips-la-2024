Network Fusion for Vaccination Probability Estimation

Overview

This project integrates egocentric and sociocentric datasets to estimate vaccination probabilities across a population network. The code processes demographic data, constructs an egocentric mixing matrix, and iteratively updates vaccination probabilities using both egocentric and sociocentric information.

Required Libraries:
numpy
pandas
networkx
matplotlib
tqdm
re


NDSSL Node and Connection Data:
contact_marginal.csv: Contains sociocentric connection data between individuals.
demog_person.csv: Contains demographic data of individuals.
GraphML Network File:
g_full_union.graphml: Contains the network structure of individuals and their attributes for the egocentric mixing matrix.

How to Run the Code

Load Node and Connection Data:
Load the sociocentric datasets (contact_marginal.csv, demog_person.csv).
Initialize a vaccination status column in the nodes dataset.
Generate Egocentric Mixing Matrix:
Extract nodes and their attributes from the GraphML file.
Generate a 3D matrix representing the distribution of age, gender, and medical status across the network.
Update Vaccination Probabilities:
Rescale age data in the sociocentric dataset to match the egocentric dataset.
Calculate initial vaccination probabilities for each node based on the egocentric mixing matrix.
Utilize sociocentric connection data to iteratively update vaccination probabilities through influence propagation across the network.
Save Final Vaccination Probabilities:
Save the final node vaccination probabilities to fused_network.csv.

The code iteratively updates vaccination probabilities and checks for convergence within a specified number of iterations (max_iter).
