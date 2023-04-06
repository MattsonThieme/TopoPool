
from typing import Optional, Tuple, Union, List
import torch
from torch import Tensor
from torch_scatter import scatter
from torch_geometric.utils import scatter, softmax, coalesce
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import global_mean_pool as gap
import torch.nn as nn
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


class TopoPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        max_clusters = None):

        super().__init__()
        self.mapper = nn.Linear(in_features=in_channels * 2, out_features=1, bias=True).to(cuda)
        self.max_clusters = max_clusters

    def reset_parameters(self):
        torch.nn.init.xavier_uniform(self.mapper.weight)
        torch.nn.init.zeros_(self.mapper.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        self.num_nodes = x.shape[0]

        if batch is None:
            self.batch = edge_index.new_zeros(x.size(0))

        elevation    = self.get_elevations(x, batch)
        cluster_info = self.get_clusters(x.to(cpu), 
                                         edge_index.to(cpu), 
                                         elevation.to(cpu), 
                                         batch.to(cpu))
        
        x, edge_index, batch, top_perm, elevation = cluster_info

        # Not implemented yet
        edge_attr = None

        return x.to(cuda), edge_index.to(cuda), edge_attr, batch.to(cuda), top_perm, elevation

    def get_clusters(
        self, 
        x, 
        edge_index, 
        elevation, 
        batch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Cannot pool a single node
        if x.shape[0] == 1:
            return x, edge_index, batch, None, elevation

        pooled_nodes = torch.tensor([], dtype=torch.long)
        pooled_batch = torch.tensor([], dtype=torch.long)
        pooled_edges = edge_index.clone()
        pool_ids     = torch.tensor([], dtype=torch.long)

        filtered_elevation = elevation.clone()
        visited_nodes      = torch.full(elevation.shape, False)
        top_perm           = torch.tensor([])
        total_pools        = 0
        num_clusters       = 0

        with torch.no_grad():
            while not visited_nodes.all():

                # Get initial cluster peaks
                peak_info = self.get_peaks(filtered_elevation, 
                                        visited_nodes, 
                                        batch)
                
                filtered_elevation, peak_indices, node_assignment = peak_info

                # Cluster all the nodes under these peaks
                clustered_nodes, batch_of_origin = self.recurse(filtered_elevation, 
                                                                peak_indices, 
                                                                edge_index, 
                                                                batch,
                                                                visited_nodes=torch.tensor([]))

                # Create new pool ID's for the clusters
                new_pool_ids = self.get_pool_ids(total_pools, batch_of_origin)
                
                # Update the variables holding pool information
                trackers = self.update_trackers(pooled_nodes,
                                                pooled_batch,
                                                pooled_edges,
                                                edge_index,
                                                pool_ids,
                                                new_pool_ids,
                                                clustered_nodes, 
                                                batch_of_origin, 
                                                visited_nodes)
                
                pooled_edges, pooled_nodes, pooled_batch, pool_ids, visited_nodes = trackers

                # Reset for the next batch
                batch_info = self.reset_batch(total_pools,
                                              new_pool_ids,
                                              num_clusters, 
                                              top_perm, 
                                              clustered_nodes)
                
                total_pools, num_clusters, top_perm = batch_info
                
                # Exit if max_clusters is defined
                if self.max_clusters != None:
                    if num_clusters == self.max_clusters:
                        break

        # Generate pooled nodes and edges
        pooled_representation = self.agg_scale_connect(x, 
                                                       pooled_nodes, 
                                                       pooled_edges, 
                                                       pooled_batch, 
                                                       elevation, 
                                                       pool_ids)
        
        scaled_x, pooled_edge_index, cluster_batches = pooled_representation
        
        return scaled_x, pooled_edge_index, cluster_batches, top_perm, elevation

    def agg_scale_connect(
        self, 
        x, 
        pooled_nodes, 
        pooled_edges, 
        pooled_batch, 
        elevation, 
        pool_ids
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Coalesce edge indices
        pooled_edge_index = coalesce(pooled_edges)

        # Cluster individual nodes
        max_pools  = scatter(src=x[pooled_nodes], index=pool_ids, dim=0, reduce='max')
        pooled_x = max_pools

        # Scale them by the softmax of their max elevations
        min_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='min')
        max_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='max')
        prominences     = max_elevations - min_elevations
        summits         = max_elevations
        promits         = prominences + summits
        cluster_batches = scatter(src=pooled_batch, index=pool_ids, dim=0, reduce='max')
        normed_promits    = softmax(promits, index=cluster_batches).unsqueeze(1)
        scaled_x        = pooled_x * normed_promits

        return scaled_x, pooled_edge_index, cluster_batches

    def reset_batch(self, 
                    total_pools, 
                    new_pool_ids,
                    num_clusters, 
                    top_perm, 
                    clustered_nodes
        ) -> Tuple[Tensor, Tensor, Tensor]:
        
        total_pools  = new_pool_ids.max() + 1
        num_clusters += 1

        # Top pool
        if top_perm.numel() == 0:
            top_perm = clustered_nodes.clone()
        
        return total_pools, num_clusters, top_perm

    def get_pool_ids(
        self, 
        total_pools, 
        batch_of_origin
    ) -> Tensor:

        # This involves shifting the label of the pool ID down so that we dont have
        # empty spaces in the final cluster_batches vector (which are all filled with zero by default)

        unique_pool_ids    = batch_of_origin.unique()
        collapsed_pool_ids = torch.range(0, unique_pool_ids.shape[0] - 1, 1).long()
        pool_id_mapping    = {u.item(): collapsed_pool_ids[i].item() for i, u in enumerate(unique_pool_ids)}
        new_pool_ids       = torch.tensor([pool_id_mapping[p.item()] for p in batch_of_origin]) + total_pools

        return new_pool_ids

    def get_peaks(
        self, 
        filtered_elevation, 
        visited_nodes, 
        batch
    ) -> Tuple[Tensor, Tensor, Tensor]:

        filtered_elevation[visited_nodes] = -float('inf')  #  masking out nodes we've already pooled
        
        peak_per_batch           = scatter(filtered_elevation, index=batch, reduce='max')
        peaks                    = torch.index_select(peak_per_batch, 0, batch)
        peak_locations           = filtered_elevation == peaks
        filtered_peaks_locations = peak_locations * (~visited_nodes)
        peak_indices             = torch.where(filtered_peaks_locations)[0]
        node_assignment          = batch[peak_indices]
        
        filtered_elevation[visited_nodes] = float('inf')
        
        return filtered_elevation, peak_indices, node_assignment

    def update_trackers(
        self, 
        pooled_nodes, 
        pooled_batch, 
        pooled_edges, 
        edge_index, 
        pool_ids, 
        new_pool_ids, 
        clustered_nodes, 
        batch_of_origin, 
        visited_nodes
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        pooled_edges = self.update_edges(pooled_edges,
                                         clustered_nodes,
                                         new_pool_ids, 
                                         edge_index)

        pooled_nodes = torch.cat((pooled_nodes, clustered_nodes.clone()))
        pooled_batch = torch.cat((pooled_batch, batch_of_origin.clone()))
        pool_ids     = torch.cat((pool_ids, new_pool_ids.clone()))
        
        visited_nodes[clustered_nodes] = True

        return pooled_edges, pooled_nodes, pooled_batch, pool_ids, visited_nodes

    def update_edges(
        self, 
        pooled_edges, 
        clustered_nodes, 
        new_pool_ids, 
        edge_index
    ) -> Tensor:

        # Get indices 
        # https://discuss.pytorch.org/t/map-values-of-a-tensor-to-their-indices-in-another-tensor/107514
        
        values = clustered_nodes

        for i in range(2):
            
            target = edge_index[i]

            t_size   = target.numel()
            v_size   = values.numel()
            t_expand = target.unsqueeze(1).expand(t_size, v_size) 
            v_expand = values.unsqueeze(0).expand(t_size, v_size)

            result            = (t_expand - v_expand == 0).nonzero()[:,1]
            expanded_pool_ids = torch.index_select(new_pool_ids, 0, result)
            mask              = torch.isin(target, clustered_nodes)

            pooled_edges[i][mask] = expanded_pool_ids
        
        return pooled_edges

    def recurse(
        self, 
        elevation, 
        node_indices, 
        edge_index, 
        batch, 
        visited_nodes
    ) -> Tuple[Tensor, Tensor]:

        # Find neighbors
        neighbors, sources, subg_edge_index, edge_mask = self.k_hop_subgraph(node_idx=node_indices, 
                                                                            num_hops=1, 
                                                                            edge_index=edge_index, 
                                                                            num_nodes=self.num_nodes)
        just_neighbors   = ~torch.isin(neighbors, node_indices)
        neighbor_indices = neighbors[just_neighbors]
        node_of_origin  = sources[just_neighbors]

        # Compare neighbor scores with source scores
        neighbor_scores = elevation[neighbor_indices]
        source_scores   = elevation[node_indices].index_select(0, node_of_origin)
        lower_neighbors = neighbor_indices[neighbor_scores <= source_scores].unique()

        # Make sure we don't revisit a node
        visited = torch.isin(lower_neighbors, visited_nodes)
        if visited.any():
            lower_neighbors = lower_neighbors[~visited]
        
        # Recurse on the lower neighbors
        if lower_neighbors.size()[0] != 0:

            visited_nodes   = torch.cat((node_indices, visited_nodes)).long()
            in_cluster, _   = self.recurse(elevation, lower_neighbors, edge_index, batch, visited_nodes)
            clustered_nodes = torch.cat((node_indices, in_cluster))
            batch_of_origin = batch[clustered_nodes]
            
            return clustered_nodes, batch_of_origin
        
        else:
            return node_indices, batch[node_indices]

    def get_elevations(self, x, batch) -> Tensor:

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)

        # Get the elevation and add small noise to break ties
        elevation = self.mapper(x).to(cpu)

        # On the off chance the graph has one node and is fed with a batch size of one
        if elevation.shape.numel() == 1:
            return elevation
        else:
            return elevation.squeeze()

    def k_hop_subgraph(
        self,
        node_idx: Union[int, List[int], Tensor],
        num_hops: int,
        edge_index: Tensor,
        relabel_nodes: bool = False,
        num_nodes: Optional[int] = None,
        flow: str = 'source_to_target',
        directed: bool = False
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
        sources = [node_idx]  # We also want to attribute the targets to their source

        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
            sources.append(row[edge_mask])

        # We want the target (subset) and source subsets
        subset = torch.cat(subsets)
        source = torch.cat(sources)

        # Convert the sources into relative positions for comparison
        source = torch.tensor([(node_idx == i).nonzero(as_tuple=True)[0].item() for i in source])

        node_mask.fill_(False)
        node_mask[subset] = True

        if not directed:
            edge_mask = node_mask[row] & node_mask[col]

        edge_index = edge_index[:, edge_mask]

        if relabel_nodes:
            node_idx = row.new_full((num_nodes, ), -1)
            node_idx[subset] = torch.arange(subset.size(0), device=row.device)
            edge_index = node_idx[edge_index]

        return subset, source, edge_index, edge_mask
