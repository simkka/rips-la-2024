# %%
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import re


def extract_first_number(data_string):
    if isinstance(data_string, str):  # Ensure the input is a string
        match = re.search(r'\d+', data_string)
        if match:
            return int(match.group(0))
    return None  # or return a default value if no match is found

# %%
# 1. LOAD THE NODE AND CONNECTION NDSSL DATA

pd.set_option('expand_frame_repr', False)

connections = pd.read_csv('/home/data/NDSSL/contact_marginal.csv')
nodes = pd.read_csv('/home/data/NDSSL/demog_person.csv')
nodes['Vaccination'] = "NA"
print(nodes.head())

# %%
# 2. GENERATE EGOCENTRIC MIXING MATRIX

mixing_matrix = np.zeros((6, 2, 2), dtype=int)
#G_fake = nx.read_graphml('/home/data/FluPaths/fake_flupaths.graphml')
G_data = nx.read_graphml('/home/data/FluPaths/g_full_union.graphml')

# Extract nodes and their attributes
nodes_data = {node: data for node, data in G_data.nodes(data=True)}
df_nodes = pd.DataFrame.from_dict(nodes_data, orient='index')

# Ensure relevant columns are numeric (get rid of NA values)
df_nodes['ages'] = df_nodes['ages'].apply(extract_first_number)
df_nodes['ages'] = pd.to_numeric(df_nodes['ages'], errors='coerce')
df_nodes['sex'] = pd.to_numeric(df_nodes['sex'], errors='coerce')
df_nodes['medical'] = df_nodes['medical'].apply(extract_first_number)
df_nodes['medical'] = pd.to_numeric(df_nodes['medical'], errors='coerce')

# Filter the DataFrame to include only the ages, sex, and medical values of interest
df_filtered = df_nodes[
    (df_nodes['ages'].between(1, 6)) &
    (df_nodes['sex'].isin([1, 2])) &
    (df_nodes['medical'].isin([1, 2]))
]

# Loop through the edges and update the matrix
for _, row in df_filtered.iterrows():
    try:
        age_idx = int(row['ages']) - 1
        sex_idx = int(row['sex']) - 1
        med_idx = int(row['medical']) -1
        mixing_matrix[age_idx, sex_idx, med_idx] += 1

    except ValueError:
        print('error')
        continue

print('Mixing Matrix:')
print(mixing_matrix)

# %%
total_counts = np.sum(mixing_matrix, axis=2)
total_counts = np.where(total_counts == 0, np.nan, total_counts)
percentages = (mixing_matrix / total_counts[:, :, np.newaxis]) * 100
percentages = np.nan_to_num(percentages)

print('Proportion Matrix:')
print(percentages)


# %%
# Ensure ego + sociocentric columns (Age, Gender) have same range

def rescale_age(age, new_min=1, new_max=6):
    normalized_age = (age - old_min) / (old_max - old_min)
    scaled_age = new_min + (new_max - new_min) * normalized_age
    return round(scaled_age)

old_min, old_max = nodes['Age'].min(), nodes['Age'].max()
nodes['Age'] = nodes['Age'].apply(rescale_age)


print(f"Age range (Sociocentric Dataset): {nodes['Age'].min()}-{nodes['Age'].max()}")
print(f"Gender range (Sociocentric Dataset): {nodes['Gender'].min()}-{nodes['Gender'].max()}")

# Update vaccination status for each node

print("Calculating Initial Vaccination Status for Each Node...")
for i, row in tqdm(nodes.iterrows(), total=nodes.shape[0]):
    vax_probability = percentages[row['Age']-1, row["Gender"]-1]
    nodes.at[i, 'Vaccination'] = round(vax_probability[0] / 100, 3)

nodes.head()
#nodes.to_csv('fusion_vaccination_1.csv', index=False)

# %%
# 3. UTILIZE SOCIOCENTRIC CONNECTION DATA TO UPDATE VACCINATION ESTIMATES

alpha = 0.5
epsilon = 1e-2
max_iter = 25

def g(probability_neigh):
    return probability_neigh

# Convert connection and nodes datasets to NetworkX graph G
G_new = nx.Graph()

print("Step 1/3: adding nodes to graph...")
for _, row in tqdm(nodes.iterrows(), total=nodes.shape[0]):
    G_new.add_node(row['Person.Id'], vaccination_prob=row['Vaccination'])

print("Step 2/3: adding edges to graph...")
for _, row in tqdm(connections.iterrows(), total=connections.shape[0]):
    G_new.add_edge(row['Person.Id.1'], row['Person.Id.2'], weight=row['weight'])

# Iteration scheme for vaccination probabilities:

print("Step 3/3: updating vaccination probabilities...")
for iteration in tqdm(range(max_iter)):
    max_change = 0
    new_probabilities = {}

    for node in G_new.nodes:
        p_i = G_new.nodes[node]['vaccination_prob']
        neighbors = list(G_new.neighbors(node))

        influence_sum = sum(
            G_new[node][neighbor]['weight'] * g((G_new.nodes[neighbor]['vaccination_prob']))
            for neighbor in neighbors
        )
        weight_sum = sum(G_new[node][neighbor]['weight'] for neighbor in neighbors)

        new_p_i = alpha * p_i + (1 - alpha) * (influence_sum / weight_sum if weight_sum != 0 else 0)
        new_probabilities[node] = new_p_i

        max_change = max(max_change, abs(new_p_i - p_i))

    for node, new_p in new_probabilities.items():
        G_new.nodes[node]['vaccination_prob'] = new_p
        # print(G.nodes[node]['vaccination_prob'])

if max_change < epsilon:
    print(f'Converged after {iteration + 1} iterations.')
else:
    print(f'Did not converge within the maximum number of {max_iter} iterations.')


# %%
node_data = pd.DataFrame(
    [(node, data['vaccination_prob']) for node, data in G_new.nodes(data=True)],
    columns=['Person.Id', 'Vaccination']
)

# Save the Final DataFrame [Node_Id, Vaccination_Probability] to a CSV file
fused_network_csv = '/home/maria/fused_network.csv'
node_data.to_csv(fused_network_csv, index=False)
print("Node data saved to " + str(fused_network_csv))


