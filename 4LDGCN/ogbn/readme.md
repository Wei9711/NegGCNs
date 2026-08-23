# LDGCN on OGBN

Large-graph versions of Layer-diverse GCN for [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) and the paper citation subgraph of [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag).

```bash
pip install ogb
python LDGCN_arxiv.py --sample_rate 0.001 --num_layers 4 --runs 10 --epochs 500
python LDGCN_mag.py --sample_rate 0.001 --num_layers 4 --runs 5 --epochs 400
```

The first run builds a shortest-path candidate cache under `cache/` (this can take a while). Later runs reuse that file. Add `--save` to write training logs under `Result/`.
