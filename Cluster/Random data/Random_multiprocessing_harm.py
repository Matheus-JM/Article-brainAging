# Biliotecas utilizadas
import numpy as np
import bct

from glob import glob
import re
import os

# Basic data manipulation and visualisation libraries
import pandas as pd

# Network Libraries
import networkx as nx

import time
from multiprocessing import Pool


def average_degree(G):
    hist = list(nx.degree(G))
    soma=0
    for i in range(len(hist)):
        soma += hist[i][1]
        
    return soma/len(G)

def shannon(A):
    S = 0
    for i in range(len(A)):
        if A[i]!=0:
            S += A[i]*np.log(1/A[i])
    
    return S

# Calculate the distance matrix

def geo(G):
    size = len(G)
    Z = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            Z[i][j] = nx.shortest_path_length(G, source = i, target = j)
        
    return Z

def entropy_geo_func(G):
    Z = geo(G)
    entropy = []
    size = len(Z) - 1
    bins = int(Z.max()+1)
    
    for j in range(len(Z)):
        hist,bin_edge = np.histogram(Z[j],bins = bins)
        entropy.append(shannon(hist/size))

    average = np.average(entropy)
    stddev = np.std(entropy)
    
    return average, stddev

def entropy_degree_func(G):
    
    # Histogram of each degree value of the network
    hist = nx.degree_histogram(G)
    # Histogram normalization
    hist[:] = [float(i) / len(G) for i in hist]

    return shannon(hist)

def denst(d, i, DIAGNOSTIC=False):
    """Creating a binarized graph with a specific density
    
    Parameters
    ---------   
    d: float
        density value
        
    i: numpy matrix
        connectivity matrix
        
    Returns
    -------
    finaldensity: float
        final density value 
    
    G1: networkx graph
        graph with the specified density
        
    """
    
    np.fill_diagonal(i, 0)
    temp = sorted(i.ravel(), reverse=True) # Will flatten it and rank corr values.
    size = len(i)
    cutoff = np.ceil(d*(size*(size-1)))
    tre = temp[int(cutoff)]
    G0 = nx.from_numpy_array(i)
    G0.remove_edges_from(list(nx.selfloop_edges(G0)))
    G1 = nx.from_numpy_array(i)
    for u,v,a in G0.edges(data=True):
        if (a.get('weight')) <= tre:
            G1.remove_edge(u, v)
    finaldensity = nx.density(G1)
    if DIAGNOSTIC == True:
        print(finaldensity)
    
    return G1

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

#Código principal com o uso da densidade

def calculation(args):
    start_t = time.perf_counter()

    nomes = ['codes']
    entropy_degree = ['entropy degree']
    entropy_geo = ['entropy geo']
    std_geo = ['std geo']
    av_degree = ['degree']
    clustering = ['clustering']
    density_list = ['density']
    average_distance = ['av. distance']
    diameter_list = ['diameter']
    
    file, atlas, condition = args

    # Separates the code of the individual
    nome_arquivo = os.path.basename(file)
    nome_sem_extensao = os.path.splitext(nome_arquivo)[0]
    codigo_nome = nome_sem_extensao.split("_")[0]
            
    print("arquivo", codigo_nome, atlas, condition, flush=True)
    
    data = pd.read_csv(file, header = None, delim_whitespace=True)
    SC_values = pd.read_csv("./random_harm/Atlas regions/" + atlas + "_subnet_order_names.txt",delim_whitespace=True, header=None)
    data = harmonization(data, SC_values)
    
    matrix = data.to_numpy()
    matrix = np.absolute(matrix)
    density = 1
    G = denst(density,matrix)

    while(nx.is_connected(G)):
        # Randomize if != 1
        if density != 1:
            matrix_random = nx.to_numpy_array(G)                #Converte a rede do networkx para um numpy array
            randomized,counter = bct.reference.randmio_und_connected(matrix_random,1) #Randomiza a rede
            G = nx.from_numpy_array(randomized)                 #Transforma a matriz randomizada em rede networkx
            
        nomes.append(codigo_nome)                                        #salvar o código do individuo
        entropy_degree.append(entropy_degree_func(G))
        a,b = entropy_geo_func(G)
        entropy_geo.append(a)
        std_geo.append(b)
        av_degree.append(average_degree(G))
        clustering.append(nx.average_clustering(G))
        density_list.append(density)
        average_distance.append(nx.average_shortest_path_length(G))
        diameter_list.append(nx.diameter(G))
                
        density -= 0.01
        G = denst(density,matrix)

    
    dados = np.array(list(zip(nomes, entropy_degree, entropy_geo, std_geo, av_degree, clustering,
                              average_distance, diameter_list, density_list)))
    name_save = './results/random_harm' + codigo_nome + '_Entropy values random_' + condition + '_' + atlas + '.txt'
    np.savetxt(name_save, dados, fmt='%s', delimiter=',')
    end_t = time.perf_counter()

    return codigo_nome, end_t - start_t


if __name__ == '__main__':
    start_t = time.perf_counter()

    path = './dataset-HCP' 
    atlas = ['fsaverage.BN_Atlas.32k_fs_LR_246regions','GlasserFreesurfer','Gordon333.32k_fs_LR_Tian_Subcortex_S1_3T']
    condition = ['HCPAging','HCPYoungAdult']

    # Escolher o atlas e a condição
    #used_atlas = atlas[0]
    #used_condition = condition[0]

    args_matrix = []
    for used_condition in condition:
        for used_atlas in atlas:
            
            filename = os.path.join(path, used_condition, '**', 'rfMRI_REST1', used_atlas, '*' + used_atlas + '_connmatrix.txt') 
            files = glob(filename, recursive = True)      #importar o nome de cada arquivo de um mapa específico
            args_aux = [(file, used_atlas, used_condition) for file in files]
            args_matrix.append(args_aux)

    args = [] # Takes the matrix formed by blocks with the altas and condition and flattens it, 
    # making each index a specific individual from a atlas and a condition
    for block in args_matrix:
        for individual in block:
            args.append(individual)
        
    
    print("Starting the processing: ")
    with Pool(8) as pool:

        results = pool.imap(calculation, args)

        for filename, duration in results:
            print(f"{filename} completed in {duration:.2f}s", flush=True)
    
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"etl took {total_duration:.2f}s total")

    