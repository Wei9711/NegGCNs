# Negative Samples-enhanced Graph Convolutional Neural Networks
Graph Convolutional Neural Networks (GCNs) have been generally accepted to be an effective tool for node representations learning. An interesting way to understand GCNs is to think of them as a message passing mechanism where each node updates its representation by accepting information from its neighbours (also known as positive samples). However, beyond these neighbouring nodes, graphs have a large, dark, all-but forgotten world in which we find the non-neighbouring nodes (negative samples).

This repository is the official implementation of several Negative Samples-enhanced Graph Convolutional Neural Networks, which can be run directly on Google colab.

# 1. NegGCN (MCGCN) 
The first method is based on Monte Carlo chains. We propose a Negative Samples-enhanced Graph Convolutional Neural Networks (NegGCNs), where the negatively sampled nodes are directly incorporated into the message passing mechanism and used to update new node feature vectors. 

![NegGcn](NegGCN(MCGCN).jpg)

Mechanism of the negative sampling graph convolution}. The central node is $v=5$ and $f(\cdot)$ is graph convolution layer\cite{kipf2016semi}. Node 4, 7 are directly linked with node 5 by real positive edges, thus positive sampling convolution is performed by $x_{pos}=f(4,7,5)$. Node 3, 8 are negative sampled using MCNS methods, which based on Markov chain MonteCarlo methods and DFS, message passing to central node $v=5$ along virtual imaginary edges, then negative sampling convolution is performed by $x_{neg}=f(3,8)$. Given a certain negative rate $\beta$, we get negative sampling graph convolution result of this layer, i.e. $x^{'} = x_{pos} -\beta x_{neg} $ 

# 2. D2GCN (DPP-DFS-GCN) 
The second method is based on  determinant point process (DPP). DPP helpl to get good negative samples, which contribute negative information to the give node contrast to its positive samples and include as much information as possible to reflect the variety of the dark world.

This work "Learning from the Dark: Boosting Graph Convolutional Neural Networks with Diverse Negative Samples" appeared in 36-th AAAI Conference on Artificial Intelligence (AAAI 2022)

![D2GCN](D2GCN-DFS.png)

The concept of DPP-based negative sampling. The target node is Node 1. Nodes 2, 3 and 4 are positive samples. Nodes 5-18 are the dark world of Node 1. The 4-length DFS path of Node 1 is $\{3, 5, 11, 13\}$, where $\{5, 11, 13\}$ are the central nodes on the path in the dark world. With their first-order neighbouring nodes, they form the candidate set of DPPs, i.e.$\{5,6,7,11,12,13,14,18\}$. The selected negative samples from this set are 6, 11, and 18, which can be seen as virtual negative links to Node 1.
