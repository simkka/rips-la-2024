Network Fusion for Vaccination Probability Estimation

Overview

This project integrates egocentric and sociocentric datasets to estimate vaccination probabilities across a population network. The code processes demographic data, constructs an egocentric mixing matrix, and iteratively updates vaccination probabilities using both egocentric and sociocentric information.

Required Libraries

Make sure the following libraries are installed:

numpy
pandas
networkx
matplotlib
tqdm
re
You can install them using:

bash
Copy code
pip install numpy pandas networkx matplotlib tqdm
Data Sources

The code uses three primary data files:

NDSSL Node and Connection Data:
contact_marginal.csv: Contains sociocentric connection data between individuals.
demog_person.csv: Contains demographic data of individuals.
GraphML Network File:
g_full_union.graphml: Contains the network structure of individuals and their attributes for constructing the egocentric mixing matrix.
How to Run the Code

To execute this code, follow these steps:

1. Load Node and Connection Data
Load the sociocentric datasets (contact_marginal.csv, demog_person.csv).
Initialize a vaccination status column in the nodes dataset.
2. Generate Egocentric Mixing Matrix
Extract nodes and their attributes from the GraphML file.
Generate a 3D matrix representing the distribution of age, gender, and medical status across the network.
3. Update Vaccination Probabilities
Rescale age data in the sociocentric dataset to match the egocentric dataset.
Calculate initial vaccination probabilities for each node based on the egocentric mixing matrix.
Utilize sociocentric connection data to iteratively update vaccination probabilities through influence propagation across the network.
4. Save Final Vaccination Probabilities
Save the final node vaccination probabilities to fused_network.csv.
The code iteratively updates vaccination probabilities and checks for convergence within a specified number of iterations (max_iter).

Example Usage

After setting up your environment and ensuring paths are correct, run the script:

bash
Copy code
python your_script.py
The final results, including vaccination probabilities for each individual, will be saved to fused_network.csv.

Notes

Ensure that the paths to the data files are set correctly before running the code.
The iterative update scheme for vaccination probabilities will stop when convergence is reached or after a maximum number of iterations.
Contributing

Contributions are welcome! Feel free to submit a pull request or report any issues.
