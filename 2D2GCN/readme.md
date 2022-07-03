# D2GCN
————————————————————

This is a PyTorch implementation of:

Learning from the Dark: Boosting Graph Convolutional Neural Networks wit Diverse Negative Samples.

Accepted by AAAI-2022

————————————————————

This code can be run directly on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1jYKH9KaozAewr56ttqxB3I_NsWLUKhnM/view?usp=sharing)

————————————————————

Introduction:

1 GCN  : Base model

2 D2GCN(DPP-DFS-GCN): Our model.  Get negative samples based on DPP and DFS

2 RGCN: Get negative samples by randomly selecting nodes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G4QDEsuTBpBkBrdpuI_ElJVHwFQ0QXVY?usp=sharing)

Reference:[HOW TO FIND YOUR FRIENDLY NEIGHBORHOOD: GRAPH ATTENTION DESIGN WITH SELF-SUPERVISION](https://openreview.net/forum?id=Wi5KUNlqWty)

3 PGCN: Get negative samples based on personalized page rank
             
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12nAPv-TmaXFgri1aT620glHoNXRvOgXO?usp=sharing)

Reference:[Understanding Negative Sampling in Graph Representation Learning](https://arxiv.org/pdf/2005.09863.pdf)
              
4 MCGCN : Get negative samples based on Monte Carko chains

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HuIDqZHDZutKRzwH4menhgYVEQsjun9a?usp=sharing)

Reference:[Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/pdf/1806.01973.pdf)
              

————————————————————

Notice:
Since DPP is a sampling procedure, the difference in time between using GPU and CPU is not significant
