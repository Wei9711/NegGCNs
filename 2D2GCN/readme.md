# D2GCN
————————————————————

This is a PyTorch implementation of:

Learning from the Dark: Boosting Graph Convolutional Neural Networks wit Diverse Negative Samples.

Accepted by AAAI-2022

————————————————————

This code can be run directly on Google Colab using CPU

————————————————————

Introduction:

1 GCN  : Base model

2 D2GCN(DPP-DFS-GCN): Our model.  Get negative samples based on DPP and DFS

2 RGCN: Get negative samples by randomly selecting nodes

Reference:[HOW TO FIND YOUR FRIENDLY NEIGHBORHOOD: GRAPH ATTENTION DESIGN WITH SELF-SUPERVISION](https://openreview.net/forum?id=Wi5KUNlqWty)

3 PGCN: Get negative samples based on personalized page rank
             
Reference:[Understanding Negative Sampling in Graph Representation Learning](https://arxiv.org/pdf/2005.09863.pdf)
              
4 MCGCN : Get negative samples based on Monte Carko chains
              
Reference:[Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973.pdf)
              

————————————————————

Notice:
Since DPP sampling uses numpy data types, the difference in time between using GPU and CPU is not significant
