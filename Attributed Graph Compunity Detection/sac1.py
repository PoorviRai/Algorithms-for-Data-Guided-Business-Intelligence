import pandas as pd
import numpy as np
import sys
from igraph import *
from scipy import spatial

# Matrices for storing the cosine similarity between vertices.
simMatrix = []
simMatrix2 = []

# For a graph, and vertex ids, returns cosine similarity
def simA(v1, v2, g):
	val1 = g.vs[v1].attributes().values()
	val2 = g.vs[v2].attributes().values()
	return 1-spatial.distance.cosine(val1, val2)  

# Input: graph g, alpha, and initial clusters assignments.
def phase1(g, alpha, C):
    V = len(g.vs)
    iter = 0
    flag = 0
	
    while (flag == 0 and iter < 15):
        flag = 1
        for i in range(V):
            maxV = -1
            maxDeltaQ = 0.0
            clusters = list(set(C))
            for j in clusters:
                  if (C[i] != j):
                      dQ = DeltaQ(alpha, C, g, i, j)
                      if (dQ > maxDeltaQ):
                          maxDeltaQ = dQ
                          maxV = j
            
            if (maxDeltaQ > 0.0 and maxV != -1):
                flag = 0
                C[i] = maxV
                
        iter + 1
        
    return C

# Implements phase2 of the algorithm
def phase2 (g, C):
	newC = seqClusters(C)
	temp = list(Clustering(newC))
	l = len(set(newC))
	simMatrix = np.zeros((l,l))
	
	for i in range(l):
		for j in range(l):
			similarity = 0.0
			for k in temp[i]:
				for m in temp[j]:
					similarity = similarity + simMatrix2[k][m]
			simMatrix[i][j] = similarity
	
	g.contract_vertices(newC)
	g.simplify(combine_edges=sum)
	return

# makes the clusters sequential. For example, cluster assignments - [2, 2, 4, 4, 5] will become: [0, 0, 1, 1, 2]
def seqClusters(C):
	mapping = {}
	newC = []
	t = 0
	for i in C:
		if i in mapping:
			newC.append(mapping[i])
		else:
			newC.append(t)
			mapping[i] = t
			t = t + 1
	return newC

# Calculates change in modularity of graph when v1 is moved to the cluster of v2
def DeltaQNew(C, g, v1, v2):
	Q1 = g.modularity(C, weights='weight')
	temp = C[v1]
	C[v1] = v2
	Q2 = g.modularity(C, weights='weight')
	C[v1] = temp
	return (Q2-Q1);

# Calculates the change in attribute similarity for the cluster of v2 after addition of v1
# Normalization by dividing it by the number of clusters and number of items in that paritulcar cluster
def DeltaQAttr(C, g, v1, v2):
	S = 0.0;
	indices = [i for i, x in enumerate(C) if x == v2]
	for v in indices:
		S = S + simMatrix[v1][v]
	return S/(len(indices)*len(set(C)))

# Calculates the total attribute similarity for the complete graph for a given clustering
# Normalization: for each cluster divide by cluster size. Overall, divide by number of clusters.
def QAttr(C, g):
	clusters = list(Clustering(C))
	S = 0.0
	for c in clusters:
		T = 0.0
		for v1 in c:
			for v2 in C:
				if (v1 != v2):
					T = T + simMatrix[v1][v2]
		T = T/len(c)
		S = S + T
	return S/(len(set(C)))

def compModularity(g, C):
	return g.modularity(C, weights='weight') + QAttr(C, g)

# Calculates overall change in modularity (structural and attribute based) by changing the 
# cluster of vertex v1 to cluster of vertex v2
def DeltaQ(alpha, C, g, v1, v2):
	d1 = DeltaQNew(C, g, v1, v2)
	d2 = DeltaQAttr(C, g, v1, v2)
	return (alpha*d1) + ((1-alpha)*d2)

# writes the clusters to the file
def writeToFile(clusters, alpha):
    if alpha == 0.5:
        file = open("communities_5.txt", 'w+')
    elif alpha == 0.0:
        file = open("communities_0.txt", 'w+')
    elif alpha == 1.0:
        file = open("communities_1.txt", 'w+')
    
    for c in clusters:
        for i in range(len(c)-1):
            file.write("%s," % c[i])
        file.write(str(c[-1]))
        file.write('\n')
    file.close()


def main(alpha):
	# Reads required files and creates an object of igraph
	attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')

	V = len(attributes)

	with open('data/fb_caltech_small_edgelist.txt') as f:
		edges = f.readlines()
	edges = [tuple([int(x) for x in line.strip().split(" ")]) for line in edges]

	g = Graph()
	g.add_vertices(V)
	g.add_edges(edges)
	g.es['weight'] = [1]*len(edges)

	for col in attributes.keys():
		g.vs[col] = attributes[col]

	# Pre-Computing Similarity Matrix
	global simMatrix
	global simMatrix2
	simMatrix = np.zeros((V,V))
	for i in range(V):
		for j in range(V):
			simMatrix[i][j] = simA(i, j, g)

	# Create a copy. 
	simMatrix2 = np.array(simMatrix)

	# Run the Algorithm
	V = g.vcount()
	print (V)
	C = phase1(g, alpha, range(V))
	print('Number of Communities after Phase 1')
	print(len(set(C)))
	C = seqClusters(C)
	#Composite modularity of phase 1 clustering
	mod1 = compModularity(g, C)

	# Phase 2
	phase2(g, C)

	# Re-running Phase 1
	V = g.vcount()
	C2 = phase1(g, alpha, range(V))
	C2new = seqClusters(C2)
	clustersPhase2 = list(Clustering(C2new))
	#Composite modularity of contracted clustering
	mod2 = compModularity(g, C)

	finalC = []
	C1new = seqClusters(C)
	clustersPhase1 = list(Clustering(C1new))

	# Mapping the super clusters from phase 2 to original vertices.
	for c in clustersPhase2:
		t = []
		for v in c:
			t.extend(clustersPhase1[v])
		finalC.append(t)

	# based on composite modularity, the corresponding clusters are written to output file
	if (mod1 > mod2):
		writeToFile(clustersPhase1, str(alpha))
		print ('Phase 1 clusters have higher modularity')
		return clustersPhase1
	else:
		writeToFile(clustersPhase2, str(alpha))
		print ('Phase 2 clusters have higher modularity')
		return clustersPhase2
	

if __name__ == "__main__":
		main(float(sys.argv[1]))