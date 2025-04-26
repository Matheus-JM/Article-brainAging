#Libraries
import pandas as pd
import numpy as np
import os
from glob import glob
import gudhi as gd


def persistent_entropy(barcodes):
    """
    Computes the persistent entropy from a list of barcodes (persistence intervals)

    Args:
    barcodes (list): List of persistence intervals as [birth, death] pairs
                     (intervals with death=inf are ignored)

    Returns:
    float: Shannon entropy value quantifying the dispersion of interval lengths
           in the persistence diagram
    """

    # Extract the lengths of the bars
    lengths = [bar[1] - bar[0] for bar in barcodes if bar[1] != float('inf')]
    
    # Normalize the lengths
    total_length = sum(lengths)
    probabilities = [length / total_length for length in lengths]
    
    # Compute the Shannon entropy
    entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    return entropy


def harmonization(individual: pd.DataFrame, SC_values: pd.DataFrame) -> pd.DataFrame:
    """
    Code to remove the rows and columns that refer to subcortical areas

    Args:
        individual (pd.DataFrame): data matrix from a subject
        SC_values (pd.DataFrame): list of the brain regions

    Returns:
        pd.DataFrame: harmonized subject data without subcortical values
    """
    
    for col in individual.columns:
        if SC_values[0][col] == 'SC':
            individual = individual.drop(col,axis=1).drop(col,axis=0)
    
    return individual


names = []
all_0_entropy = []
all_1_entropy = []
all_2_entropy = []


path = './dataset-HCP'
atlas = ['fsaverage.BN_Atlas.32k_fs_LR_246regions']
#atlas = ['GlasserFreesurfer']
#atlas = ['Gordon333.32k_fs_LR_Tian_Subcortex_S1_3T']
condition = ['HCPYoungAdult']
#condition = ['HCPAging']

for a in atlas:
    for c in condition:
        filename = os.path.join(path, c, '**', 'rfMRI_REST1', a, '*' + a + '_connmatrix.txt')
        files = glob(filename, recursive=True)
        print(files)

        for f in files:
            name_path = os.path.basename(f)                                # extract filename from full path
            name_no_extens = os.path.splitext(name_path)[0]                # remove the file extension
            code_name = name_no_extens.split("_")[0]                        # take the first component of the name
            nomes.append(code_name)

            individual = pd.read_csv(file, header = None, delim_whitespace=True)
            SC_values = pd.read_csv("./random_harm/Atlas regions/" + a + "_subnet_order_names.txt",delim_whitespace=True, header=None)
            individual = harmonization(individual, SC_values)
    
            corr_matrix = individual.corr().to_numpy()
            distance_matrix = 1 - corr_matrix
    
            #Cálculo dos "barcodes"
            rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=3.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
            persistence = simplex_tree.persistence()
            
            barcode_0 = [p[1] for p in persistence if p[0] == 0]
            barcode_1 = [p[1] for p in persistence if p[0] == 1]
            barcode_2 = [p[1] for p in persistence if p[0] == 2]
            
            entropy_0 = persistent_entropy(barcode_0)
            all_0_entropy.append(entropy_0)
            entropy_1 = persistent_entropy(barcode_1)
            all_1_entropy.append(entropy_1)
            entropy_2 = persistent_entropy(barcode_2)
            all_2_entropy.append(entropy_2)
    
data = np.array(list(zip(names, all_0_entropy, all_1_entropy, all_2_entropy))) 
name_save = './results/harm' + code_name + '_persistent entropy_' + condition + '_' + atlas + '.txt'
np.savetxt(name_save, data, fmt='%s', delimiter=',')
    
    




    