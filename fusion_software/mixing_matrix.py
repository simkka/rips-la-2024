import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

# only makes matrices for egocentric

def make_mixing_matrix(attribute_1, attribute_2, include_edges=False, visualize=False):
    G = nx.read_graphml('/home/data/FluPaths/flupaths.graphml')

    attr_stats_1, attr_stats_2 = [], []
    for _, data in G.nodes(data=True):

        if f'{attribute_1}' in data and f'{attribute_2}' in data: 
            # Attribute_1 Counting + Error Handling:
            if data[f'{attribute_1}'] == 'NA' or data[f'{attribute_1}'] == 'na': 
                attr_stats_1.append(11)
            else:
                try: 
                    value = int(data[f'{attribute_1}'])
                    attr_stats_1.append(int(value))
                except ValueError:
                    print(f"Invalid value encountered: {value}")
                    attr_stats_1.append(11)

            # Attribute_2 Counting + Error Handling:
            if data[f'{attribute_2}'] == 'NA' or data[f'{attribute_2}'] == 'na': 
                attr_stats_2.append(11)
            else:
                try: 
                    value = int(data[f'{attribute_2}'])
                    attr_stats_2.append(int(value))
                except ValueError:
                    print(f"Invalid value encountered: {value}")
                    attr_stats_2.append(11)
        else:
            print(f"Attribute '{attribute_1}' or '{attribute_2}' not found in node data")
            return None
    
    set_1, set_2 = sorted(set(attr_stats_1)), sorted(set(attr_stats_2))

    mixing_matrix = pd.DataFrame(0, index=set_1, columns=set_2)
    print("Unique values in attribute 1: ", set_1)
    print("Unique values in attribute 2: ", set_2)

    for edge in tqdm(G.edges()):
        node1, node2 = edge
        
        if G.nodes[node1][attribute_1] == 'NA' or G.nodes[node1][attribute_2] == 'NA': continue

        attr1_value_1 = G.nodes[node1][attribute_1] if G.nodes[node1][attribute_1] != 'NA' else 11
        attr2_value_1 = G.nodes[node1][attribute_2] if G.nodes[node1][attribute_2] != 'NA' else 11

        attr1_value_2 = G.nodes[node2][attribute_1] if G.nodes[node2][attribute_1] != 'NA' else 11
        attr2_value_2 = G.nodes[node2][attribute_2] if G.nodes[node2][attribute_2] != 'NA' else 11

        try:
            mixing_matrix.loc[int(attr1_value_1), int(attr2_value_1)] += 1
            mixing_matrix.loc[int(attr1_value_2), int(attr2_value_2)] += 1
        except:
            print(f'Error with {node1} and/or {node2}'); continue

    if visualize:
        np.savetxt(f'/home/data/NDSSL/mixing_ego_{attribute_1.lower()}_{attribute_2.lower()}.csv', mixing_matrix, delimiter=',')
        plt.title(f'Egocentric {attribute_1.capitalize()} x {attribute_2.capitalize()} Mixing Matrix')
        plt.imshow(mixing_matrix, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Proportion of Interactions'); plt.tight_layout()
        plt.xlabel(f'{attribute_2.capitalize()} Group 2'); plt.xticks(np.arange(len(mixing_matrix[0])), mixing_matrix[0], rotation=45)
        plt.ylabel(f'{attribute_1.capitalize()} Group 1'); plt.yticks(np.arange(len(mixing_matrix[0])), mixing_matrix[0], rotation=0)
        plt.savefig(f'/home/data/NDSSL/mixing_ego_{attribute_1.lower()}_{attribute_2.lower()}.png'); plt.show()
        plt.close()

    # print(mixing_matrix)
    return mixing_matrix

mm = make_mixing_matrix(attribute_1='ages', attribute_2='medical', include_edges=False, visualize=True)
mm = make_mixing_matrix(attribute_1='sex', attribute_2='medical', include_edges=False, visualize=True)
# mm = make_mixing_matrix(attribute_1='sex', attribute_2='ages', include_edges=False, visualize=True)