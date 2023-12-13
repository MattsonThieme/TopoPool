
from typing import Optional, Tuple, Union, List
import torch
from torch import Tensor
from torch_scatter import scatter
import torch_geometric
from torch_geometric.utils import scatter, softmax, coalesce, degree, add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import SparseTensor
import torch_sparse
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


# 
class TopoPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        gen_edges = False,
        max_clusters = None):

        super().__init__()
        self.mapper = nn.Linear(in_features=in_channels, out_features=1, bias=True).to(cuda)
        self.max_clusters = max_clusters
        self.clusterer = Clusterer()
        self.gen_edges = gen_edges

    def reset_parameters(self):
        torch.nn.init.xavier_uniform(self.mapper.weight)
        torch.nn.init.zeros_(self.mapper.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple, Tensor]:

        self.num_nodes = x.shape[0]

        if batch is None:
            self.batch = edge_index.new_zeros(x.size(0))

        # Get elevations
        elevation    = self.get_elevations(x, batch)
        edge_index   = coalesce(edge_index)

        # Cluster based on the expand and descend method
        cluster_mems, cluster_ids, cluster_batch, peaks, troughs = self.clusterer(elevation.unsqueeze(1), edge_index, batch)
        
        # Scale the clusters
        pooled_x, cluster_batch, scaling = self.scale(x, elevation, cluster_mems, cluster_ids, cluster_batch)
        self.num_pools = pooled_x.shape[0]

        # Connect the pools if needed. This is made as an option to save training time if it isn't needed
        if self.gen_edges:
            edge_index = add_self_loops(edge_index)[0]
            edge_index = self.connect(edge_index, peaks)

        x = pooled_x
        batch = cluster_batch
        perms = (cluster_mems, cluster_ids, scaling)

        # Not implemented yet
        edge_attr = None

        return x.to(cuda), edge_index.to(cuda), edge_attr, batch.to(cuda), (perms, scaling), elevation

    def get_elevations(self, x, batch) -> Tensor:

        # Get the elevation and add small noise to break ties
        elevation = self.mapper(x)
        elevation = softmax(elevation, batch)

        return elevation.squeeze(-1)

    def scale(self, x, elevation, cluster_mems, cluster_ids, cluster_batch):

        max_elevations  = scatter(src=elevation[cluster_mems], index=cluster_ids, dim=0, reduce='max')
        x               = x[cluster_mems]
        max_pooled_x    = scatter(src=x, index=cluster_ids, dim=0, reduce='max')
        cluster_batch   = scatter(src=cluster_batch.float(), index=cluster_ids, dim=0, reduce='max').long()
        normed_elevs    = softmax(max_elevations, index=cluster_batch).unsqueeze(1)
        pooled_x        = max_pooled_x  * normed_elevs

        return pooled_x, cluster_batch, normed_elevs
    

    def connect(self, edge_index, peaks):

        ########### Adapted from ASAPool code, originally from DiffPool
        
        # Graph coarsening
        N = self.num_nodes
        row_initial, col_initial = edge_index[0], edge_index[1]
        A_initial = SparseTensor(row=row_initial, col=col_initial, sparse_sizes=(N, N))
        S_initial = SparseTensor(row=row_initial, col=col_initial, sparse_sizes=(N, N))

        C = peaks
        S = torch_sparse.index_select(S_initial, 1, C)
        A = torch_sparse.matmul(torch_sparse.matmul(torch_sparse.t(S), A_initial), S)

        row, col, _ = A.coo()
        coarsened_edge_index = torch.stack([row, col], dim=0).to(device)

        return coarsened_edge_index
    


class Clusterer(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super(Clusterer, self).__init__()
        pass

    # Return the list of greater-than and less-than edges
    def message(self, x_i, x_j):
        gt = x_i >= x_j
        lt = x_i <= x_j
        return gt, lt
    
    # If a node is >/< all of its neighbors, map it to peaks/troughs
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        deg     = degree(index, self.num_nodes, dtype=torch.long).unsqueeze(1)
        gt, lt  = inputs
        try:
            peaks, _   = torch.where(scatter(gt.long(), index, reduce='sum') == deg)
            troughs, _ = torch.where(scatter(lt.long(), index, reduce='sum') == deg)
        except:
            gt = gt.long().squeeze(-1)
            lt = lt.long().squeeze(-1)
            peaks, _   = torch.where(scatter(gt.long(), index, reduce='sum') == deg)
            troughs, _ = torch.where(scatter(lt.long(), index, reduce='sum') == deg)

        return peaks, troughs
    
    def update(self, aggr_out):
        return aggr_out

    def fill(self, elevation, peaks, troughs, edge_index, batch):

        # Cluster members and their ids, start with the peaks
        cluster_mems  = peaks
        cluster_ids   = peaks
        cluster_batch = batch[peaks]
        first_pass = True

        to_visit = peaks

        while to_visit.numel() > 0:

            neighbors, origins, edge_index = one_hop_subgraph(to_visit, 
                                                                 edge_index, 
                                                                 num_nodes=self.num_nodes)

            # Only move towards neighbors <= origins
            descent_mask = (elevation[neighbors] <= elevation[origins]).squeeze(1)
            neighbors = neighbors[descent_mask]
            origins = origins[descent_mask]
            
            if not first_pass:
                
                # Get mapping
                mapping = torch.cat((cluster_mems.unsqueeze(1), cluster_ids.unsqueeze(1)), dim=1)
                mapping = torch.unique(mapping, dim=0)

                # Organize
                origins, perm = torch.sort(origins)
                neighbors = neighbors[perm]

                # Repeat values
                to_repeat = torch.bincount(mapping[:,0][torch.isin(mapping[:,0], origins)])[origins]
                origins   = torch.repeat_interleave(origins, to_repeat)
                neighbors = torch.repeat_interleave(neighbors, to_repeat)

                # Expand any repeated values in origins that are not repeated in mapping
                extracted_map = mapping[torch.isin(mapping[:,0], origins)][:,0]
                origins_count = torch.bincount(origins)[extracted_map]

                # If there are duplicates in the origins, the origins_count can be wrong because it
                # counts the expanded ones multiple times. This fixes that by appropriately scaling 
                # the counts of those repeated values.
                _, ext_inv, num_attributions = extracted_map.unique(return_inverse=True, return_counts=True)
                _, _, num_appearances  = origins.unique(return_inverse=True, return_counts=True)
                rescaled_counts = (num_appearances / num_attributions).long()
                origins_count = torch.index_select(rescaled_counts, 0, ext_inv)

                # Extract the origins
                extracted_origins = mapping[torch.isin(mapping[:,0], origins)][:,1]

                # Expand the origins where we need to
                origins = torch.repeat_interleave(extracted_origins, origins_count)

            # First-come first-served: if multiple clusters reach the same node at the same time, they can all claim it
            unvisited = ~torch.isin(neighbors, cluster_mems)

            # Move forward to the unvisited neighbors
            to_visit     = neighbors[unvisited]
            origins      = origins[unvisited]

            # Drop duplicates
            # TODO: Should use torch coalesce here, probably faster
            merged   = torch.cat((to_visit.unsqueeze(1), origins.unsqueeze(1)), dim=1).unique(dim=0)
            to_visit = merged[:, 0]
            origins  = merged[:, 1]

            cluster_mems  = torch.cat((cluster_mems, to_visit))
            cluster_ids   = torch.cat((cluster_ids, origins))
            cluster_batch = torch.cat((cluster_batch, batch[origins]))

            first_pass = False
        
        # Remove duplicates
        merged        = torch.cat((cluster_mems.unsqueeze(1), cluster_ids.unsqueeze(1), cluster_batch.unsqueeze(1)), dim=1).unique(dim=0)
        cluster_mems  = merged[:, 0]
        cluster_ids   = merged[:, 1]
        cluster_batch = merged[:, 2]

        if cluster_ids.unique().numel() != peaks.unique().numel():
            print("Cluster ids not equal to peaks")

        # Collapse the cluster IDs into contiguous values in [0, num_clusters]
        unique_pools = cluster_ids.unique()
        new_pool_ids = torch.arange(unique_pools.size(0), device=cluster_ids.device)
        map_store = torch.zeros(cluster_ids.max() + 1).long().to(device)
        map_store[cluster_ids.unique()] = new_pool_ids
        cluster_ids = map_store[cluster_ids]
        
        return cluster_mems, cluster_ids, cluster_batch, peaks
    
    def forward(self, elevation, edge_index, batch):
        self.num_nodes = elevation.size(0)
        edge_index    = coalesce(edge_index)

        # Get initial candidate peaks
        # edge_index = add_self_loops(edge_index, num_nodes = self.num_nodes)[0]
        peaks, troughs = self.propagate(x=elevation, edge_index=edge_index)
        peaks = peaks.unique()

        # If there is only a single node in molecule, removing the self loops removes that node
        included_peaks = torch.isin(peaks, edge_index)
        if not included_peaks.all():
            add_back = peaks[~included_peaks]
            add_back = torch.cat((add_back.unsqueeze(0), add_back.unsqueeze(0)))
            edge_index = torch.cat((edge_index, add_back), dim=1)

        # Fill clusters
        cluster_mems, cluster_ids, cluster_batch, peaks = self.fill(elevation, peaks, troughs, edge_index, batch)

        return cluster_mems, cluster_ids, cluster_batch, peaks, troughs


def one_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    node_mask.fill_(False)
    node_mask[subsets[-1]] = True
    torch.index_select(node_mask, 0, row, out=edge_mask)
    subset = col[edge_mask]

    # Split edge index into visited and unvisited edges
    # The default behavior of this function was to extract every 1-hop
    # neighbor from edge_index, which is not what we want because 
    # those neighbors may be connected to other more distant nodes
    
    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    
    # Filter for sources in the original node_idx set (the nodes we're hopping from)
    src_mask = torch.isin(edge_index[0], node_idx)

    # Attribute each 1-hop neighbor to their node of origin
    # These will be used to assign cluster labels
    origins   = edge_index[0][src_mask]
    neighbors = edge_index[1][src_mask]
    
    return neighbors, origins, edge_index
