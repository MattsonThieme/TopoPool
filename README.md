# Topographic Pooling

This repo contains the code and data for the following paper:


- TopoPool: An Adaptive Graph Pooling Layer for Extracting Molecular and Protein Substructures. Thieme, et al., NeurIPS 2023 Workshop on New Frontiers of AI for Drug Discovery and Development (AI4D3 2023) [[PDF](https://ai4d3.github.io/papers/21.pdf)]


## Setup

To setup the conda environment for the training, run:

```bash
$: bash setup.sh
```

## Benchmarking

Running the following will initiate the full benchmarking script:

```bash
$: python benchmarker.py
```


## Topographic Pooling at a glance

This is a max-*region* pooling operation.

Graph pooling methods implement the following broad structure [(Grattarola, 2021)](https://arxiv.org/abs/2110.05292):

- **Select**: using some thresholding mechanism, select a subset of nodes
- **Reduce**: aggregate the selected nodes
- **Connect**: produce a new connectivity structure between the aggregated representations

Here, we propose a chemistry-inspired **Select** operation that respects the following fact: 

- A molecules' bioactivity is often heavily influenced by the discrete, contiguous substructures (functional groups, etc) from which it is composed.

Intuitively, the proposed algorithm operates as follows:

1. Transform node *features*, $x_i \in \mathbb{R}^F$, into contextualized node *representations*, $h_i \in \mathbb{R}^d$, using a standard GNN.
2. Using a dense *scoring-layer*, $S_{\theta}(\cdot)$, generate scalar-valued scores $s_i = S_{\theta}(h_i) \in \mathbb{R}^1, s_i  \geq  0$ for each node (atom).
3. We now think of the scores + connectivity structure as defining a *topography* or *contour map*, like a mountain range, over the graph.
4. Beginning from the node $i$ with the highest score, we *descend* down all possible paths, defined by the connectivity structure, until we reach *the bottom of a valley* (any valley, where the next neighbor's score is higher than the current node's).
5. Each node reached using this procedure - descending from node with the maximum score - is assigned to one cluster
6. Using the remaining, unassigned nodes, we repeat the above process until all nodes have been assigned to a cluster. See Figure \ref{flow}.

Advantages: 

- This implicitly ensures that the pooling operation chooses contiguous subgraphs.
- Principled approach for bounding clusters: obviates the need for a pooling `threshold' as it uses the naturally imposed *topographic structure (score values + connectivity) of the graph.
- Allows the model to learn an arbitrary number of clusters of arbitrary size, obviating the need to define the maximum number of clusters.
- Parallelizable, works over the batch dimension for batched graphs.
