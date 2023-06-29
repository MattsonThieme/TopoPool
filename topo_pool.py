from typing import Optional, Tuple, Union, List
import torch
from torch import Tensor
from torch_scatter import scatter, scatter_std
import torch_geometric
from torch_geometric.utils import scatter, softmax, coalesce, degree, remove_self_loops, add_self_loops, to_undirected, get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GraphNorm, LayerNorm, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.typing import SparseTensor
import torch_sparse
import torch.nn as nn
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

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

    # Get a mask going each way, adjoining the nodes in node_idx and subset
    src_node_idx_edge_mask = torch.isin(edge_index[0], node_idx)
    
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

class Clusterer_MT(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super(Clusterer_MT, self).__init__()
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
        peaks, _   = torch.where(scatter(gt.long(), index, reduce='sum') == deg)
        troughs, _ = torch.where(scatter(lt.long(), index, reduce='sum') == deg)

        return peaks, troughs
    
    def update(self, aggr_out):
        return aggr_out

    def fill(self, elevation, peaks, troughs, edge_index, batch):

        edge_index = remove_self_loops(edge_index)[0]

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
                mapping_count = torch.bincount(mapping[:, 0])[extracted_map]

                # If there are duplicates in the origins, the origins_count can be wrong because it
                # counts the expanded ones multiple times. This fixes that by appropriately scaling 
                # the counts of those repeated values.
                _, ext_inv, num_attributions = extracted_map.unique(return_inverse=True, return_counts=True)
                _, _, num_appearances  = origins.unique(return_inverse=True, return_counts=True)
                rescaled_counts = (num_appearances / num_attributions).long()
                origins_count = torch.index_select(rescaled_counts, 0, ext_inv)

                # Extract the origins
                extracted_origins = mapping[torch.isin(mapping[:,0], origins)][:,1]

                if torch.repeat_interleave(extracted_origins, origins_count).shape != neighbors.shape:
                    raise Exception
                    # This happens when we have the same origin multiple times. It is then expanded and
                    # multi-counted when we do origins_count again.

                # Expand the origins where we need to
                origins = torch.repeat_interleave(extracted_origins, origins_count)

            '''
            # Don't let clusters revisit nodes
            attributed   = torch.cat((cluster_mems.unsqueeze(1), cluster_ids.unsqueeze(1)), dim=1)
            neighb_label = torch.cat((neighbors.unsqueeze(1), origins.unsqueeze(1)), dim=1)
            
            equality     = torch.eq(attributed[:, None], neighb_label)
            row_equality = torch.all(equality, dim=2)
            unvisited    = ~torch.any(row_equality, dim=0)
            '''

            # First-come first-served: if multiple clusters reach the same node at the same time, they can all claim it
            unvisited = ~torch.isin(neighbors, cluster_mems)

            # Move forward to the unvisited neighbors
            to_visit     = neighbors[unvisited]
            origins      = origins[unvisited]

            cluster_mems  = torch.cat((cluster_mems, to_visit))
            cluster_ids   = torch.cat((cluster_ids, origins))
            cluster_batch = torch.cat((cluster_batch, batch[origins]))

            # We don't need to get the one-hop neighborhood of a trough because we know it doesn't have
            # any lower neighbors. This also prevents the multiple attribution problem and automatically
            # stops each cluster at the boundaries defined by troughs
            # trough_mask  = ~torch.isin(to_visit, troughs)
            # to_visit     = to_visit[trough_mask]

            # print(to_visit)

            first_pass = False
            

        # Collapse the cluster IDs into contiguous values in [0, num_clusters]
        unique_pools = cluster_ids.unique()
        new_pool_ids = torch.arange(unique_pools.size(0), device=cluster_ids.device)
        mapping = torch.cat((unique_pools.unsqueeze(1), new_pool_ids.unsqueeze(1)), dim=1)
        mask = cluster_ids == mapping[:, :1]
        cluster_ids = (1 - mask.sum(dim=0)) * cluster_ids + (mask * mapping[:,1:]).sum(dim=0)

        # Use the coalesce function to filter out repeated attributions - still not sure why this happens
        coalesced = coalesce(torch.cat((cluster_mems.unsqueeze(0), cluster_ids.unsqueeze(0))), cluster_batch, reduce='mean')
        cluster_mems, cluster_ids, cluster_batch = coalesced[0][0], coalesced[0][1], coalesced[1].long()

        return cluster_mems, cluster_ids, cluster_batch

    def filter_adjacent(self, x, indices, edge_index):

        # Join peak ids that are connected and have the same value
        values, counts = x[indices].unique(return_counts=True)
        if (counts != 1).any():
            eq_values = values[~(counts == 1)]
            eq_indices = torch.isin(x[indices], eq_values.unsqueeze(1))
            eq_indices = indices[eq_indices.squeeze()]

            # If any are connected by one hop, retain the smallest index (arbitrary, doesnt actually matter)
            neighbors, origins, _ = one_hop_subgraph(eq_indices, edge_index, num_nodes=self.num_nodes)
            eq_neighbors = neighbors[torch.isin(neighbors, eq_indices)]
            eq_indices = torch.unique(x[eq_neighbors], return_inverse=True)[1]
            drop_indices = scatter(eq_neighbors.float(), eq_indices.squeeze(), reduce="max").long()
            indices = indices[~torch.isin(indices, drop_indices)]

        return indices
    
    def forward(self, elevation, edge_index, batch):
        self.num_nodes = elevation.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes = self.num_nodes)

        # Get initial candidate peaks
        peaks, troughs = self.propagate(x=elevation, edge_index=edge_index)
        # flats = peaks[torch.isin(peaks, troughs)]

        # peaks = peaks[~torch.isin(peaks, flats)]
        # troughs = troughs[~torch.isin(troughs, flats)]

        # Join peak/trough ids that are adjacent and have the same value
        # filtered_peaks   = self.filter_adjacent(elevation, peaks, edge_index)
        # filtered_troughs = self.filter_adjacent(elevation, troughs, edge_index)

        # Fill clusters
        cluster_mems, cluster_ids, cluster_batch = self.fill(elevation, peaks, troughs, edge_index, batch)

        return cluster_mems, cluster_ids, cluster_batch, peaks, troughs

class Smoothing(torch_geometric.nn.MessagePassing):
    def __init__(self):
        super(Smoothing, self).__init__()
        pass

    # Return the list of greater-than and less-than edges
    def message(self, x_i, x_j):
        return x_j
    
    # If a node is >/< all of its neighbors, map it to peaks/troughs
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, reduce='mean')
    
    def update(self, aggr_out):
        return aggr_out
    
    def forward(self, elevation, edge_index):
        self.num_nodes = elevation.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes = self.num_nodes)

        # Get initial candidate peaks
        smoothed_elevations = self.propagate(x=elevation, edge_index=edge_index)

        return smoothed_elevations.squeeze(1)

class TopoPool_v3(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        gen_edges = False,
        max_clusters = None):

        super().__init__()
        # self.n1 = LayerNorm(in_channels)
        self.mapper = nn.Linear(in_features=in_channels, out_features=1, bias=True).to(cuda)
        # self.n2 = LayerNorm(in_channels)
        self.max_clusters = max_clusters
        self.clusterer = Clusterer_MT()
        self.smoother = Smoothing()
        self.gen_edges = gen_edges
        
        # Noise scaling
        self.std = 1.0
        self.rate = 0.98

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

        # First LayerNorm
        # x = self.n1(x, batch)

        elevation    = self.get_elevations(x, batch)

        smoothing_method = None
        if smoothing_method == 'avg':
            elevation = self.smoother(elevation.unsqueeze(1), edge_index)
        
        elif smoothing_method == 'laplacian':
            edges, _                        = add_self_loops(edge_index, num_nodes=self.num_nodes)
            lap_edge_index, lap_edge_weight = get_laplacian(edges, normalization='rw')
            adj                             = torch.sparse_coo_tensor(lap_edge_index, lap_edge_weight, [self.num_nodes, self.num_nodes])
            if len(elevation.shape) == 1:
                signal_smooth = torch.spmm(adj, elevation.unsqueeze(-1))
            else:
                signal_smooth = torch.spmm(adj, elevation)

            # Subtract from the original signal to create a low-pass filter
            elevation = elevation - signal_smooth.squeeze()
        
        else:
            pass

        ##########
        ########## New clustering test
        ##########
        # a = time.time()
        edge_index = coalesce(edge_index)
        cluster_mems, cluster_ids, cluster_batch, peaks, troughs = self.clusterer(elevation.unsqueeze(1), edge_index, batch)

        # Generate scaled pools
        pooled_x, cluster_batch, scaling = self.scale(x, elevation, cluster_mems, cluster_ids, cluster_batch)

        # Second LayerNorm
        # x = self.n2(pooled_x, cluster_batch)

        # Connect pools
        if self.gen_edges:
            edge_index = self.connect(cluster_mems, cluster_ids, batch, edge_index, peaks, troughs)

        x = pooled_x
        batch = cluster_batch
        perms = (cluster_mems, cluster_ids, scaling)

        # b = time.time()

        ##########
        ##########
        ##########
        '''
        c = time.time()
        cluster_info = self.get_clusters(x.to(cpu), 
                                         edge_index.to(cpu), 
                                         elevation.to(cpu), 
                                         batch.to(cpu))
        
        x, edge_index, batch, perms, elevation, scaling = cluster_info
        d = time.time()
        print("Old: {}".format(d - c))
        print("New: {}".format(b - a))
        '''

        # Not implemented yet
        edge_attr = None

        return x.to(cuda), edge_index.to(cuda), edge_attr, batch.to(cuda), (perms, scaling), elevation


    def get_elevations(self, x, batch) -> Tensor:

        # x_glob_avg = torch.index_select(gap(x, batch), 0, batch)
        # x_glob_max = torch.index_select(gmp(x, batch), 0, batch)
        # x_glob = x_glob_avg + x_glob_max
        # x = torch.cat((x, x_glob_avg, x_glob_max), dim=1)

        # Get the elevation and add small noise to break ties
        elevation = self.mapper(x)  # .to(cpu)

        # Push the peaks away from the troughs
        '''
        if self.training:
            means = torch.full(elevation.shape, 0.0, device=device)
            stds = scatter_std(elevation.squeeze(1).detach(), batch).to(device) / 10
            stds = torch.index_select(stds, 0, batch).unsqueeze(1)
            noise = torch.normal(means, stds)
            elevation = elevation + noise
        else:
            pass
        '''
        
        # elevation = torch.tanh(elevation)
        elevation = softmax(elevation, batch)
        
        # On the off chance the graph has one node and is fed with a batch size of one
        if elevation.shape.numel() == 1:
            return elevation
        else:
            return elevation.squeeze()

    def scale(self, x, elevation, cluster_mems, cluster_ids, cluster_batch):

        pooled_x        = scatter(src=x[cluster_mems], index=cluster_ids, dim=0, reduce='max')
        max_elevations  = scatter(src=elevation[cluster_mems], index=cluster_ids, dim=0, reduce='max')
        cluster_batch   = scatter(src=cluster_batch.float(), index=cluster_ids, dim=0, reduce='max').long()
        max_normed      = softmax(max_elevations, index=cluster_batch).unsqueeze(1)
        pooled_x        = pooled_x * max_normed

        return pooled_x, cluster_batch, max_normed

    def connect(self, cluster_mems, cluster_ids, batch, edge_index, peaks, troughs):


        ########### Adapted from ASAPool code, originally from DiffPool
        
        # Graph coarsening.
        N = self.num_nodes
        row, col = edge_index[0], edge_index[1]
        A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, sparse_sizes=(N, N))

        C = peaks
        S = torch_sparse.index_select(S, 1, C)
        A = torch_sparse.matmul(torch_sparse.matmul(torch_sparse.t(S), A), S)

        row, col, _ = A.coo()
        edge_index = torch.stack([row, col], dim=0)
        
        ###########

        '''
        members, counts = cluster_mems.unique(return_counts=True)
        if counts[counts > 1].any():
            # Find contacts
            contacts        = torch.isin(cluster_mems, members[counts > 1])
            contact_nodes   = cluster_mems[contacts]
            contact_pools   = cluster_ids[contacts]
            contact_nodes, perm = contact_nodes.sort()
            contact_pools = contact_pools[perm]
            # Split, unify
            _, split_counts = contact_nodes.unique(return_counts=True)
            neighbors = torch.split(contact_pools, split_counts.tolist())
            new_edges = torch.cat([torch.combinations(i, with_replacement=True) for i in neighbors]).T
            # Implicit coalesce in to_undirected
            edge_index = to_undirected(new_edges)
            edge_index = add_self_loops(edge_index, num_nodes=cluster_ids.max() + 1)[0]
        else:
            # Only one cluster, so no contacts
            pass
        '''

        return edge_index
    
    def get_clusters(
        self, 
        x, 
        edge_index, 
        elevation, 
        batch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Cannot pool a single node
        if x.shape[0] == 1:
            return x, edge_index, batch, None, elevation, None

        pooled_nodes = torch.tensor([], dtype=torch.long)
        pooled_batch = torch.tensor([], dtype=torch.long)
        pooled_edges = edge_index.clone()
        pool_ids     = torch.tensor([], dtype=torch.long)

        filtered_elevation = elevation.clone()
        visited_nodes      = torch.full(elevation.shape, False)
        perms           = []
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
                                              perms, 
                                              clustered_nodes)
                
                total_pools, num_clusters, perms = batch_info
                
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
        
        scaled_x, pooled_edge_index, cluster_batches, promits = pooled_representation
        
        return scaled_x, pooled_edge_index, cluster_batches, perms, elevation, promits

    def agg_scale_connect(
        self, 
        x, 
        pooled_nodes, 
        pooled_edges, 
        pooled_batch, 
        elevation, 
        pool_ids
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Scale x by elevation
        # x = x * elevation.unsqueeze(1) / elevation.unsqueeze(1).detach()

        # Coalesce edge indices
        pooled_edge_index = coalesce(pooled_edges)

        # Generate clusters based on pool_ids
        pooled_x  = scatter(src=x[pooled_nodes], index=pool_ids, dim=0, reduce='max')

        # Scale pools
        # cluster_scales = self.scaler(pooled_x)
        # scaled_x = pooled_x * cluster_scales

        # Scale them by the softmax of their max elevations
        
        # min_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='min')
        max_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='max')
        # prominences     = max_elevations - min_elevations
        summits         = max_elevations
        promits         = summits  # prominences + summits
        cluster_batches = scatter(src=pooled_batch, index=pool_ids, dim=0, reduce='max')
        normed_promits  = softmax(promits, index=cluster_batches).unsqueeze(1)
        scaled_x        = pooled_x * normed_promits

        return scaled_x, pooled_edge_index, cluster_batches, normed_promits

    def reset_batch(self, 
                    total_pools, 
                    new_pool_ids,
                    num_clusters, 
                    perms, 
                    clustered_nodes
        ) -> Tuple[Tensor, Tensor, Tensor]:
        
        total_pools  = new_pool_ids.max() + 1
        num_clusters += 1

        # Top pool
        # if perms.numel() == 0:
        #     perms = clustered_nodes.clone()

        perms.append(clustered_nodes.clone())
        
        return total_pools, num_clusters, perms

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

        # Can't recurse any further if we're only given a single node without neighbors
        if (~just_neighbors).all():
            return node_indices, batch[node_indices]
        
        neighbor_indices = neighbors[just_neighbors]
        node_of_origin  = sources[just_neighbors]

        # Compare neighbor scores with source scores
        neighbor_scores = torch.round(elevation[neighbor_indices], decimals=4)
        source_scores   = torch.round(elevation[node_indices].index_select(0, node_of_origin), decimals=4)
        lower_neighbors = neighbor_indices[neighbor_scores <= source_scores].unique()  # Try <= with rounding to third decimal place

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



class TopoPool_v2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        gen_edges = False,
        max_clusters = None):

        super().__init__()
        # self.n1 = LayerNorm(in_channels)
        self.mapper = nn.Linear(in_features=in_channels, out_features=1, bias=True).to(cuda)
        # self.n2 = LayerNorm(in_channels)
        self.max_clusters = max_clusters
        self.clusterer = Clusterer_MT()
        self.gen_edges = gen_edges
        
        # Noise scaling
        self.std = 1.0
        self.rate = 0.98

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

        # First LayerNorm
        # x = self.n1(x, batch)

        elevation    = self.get_elevations(x, batch)
        # elevation    = torch.round(elevation, decimals=5)
        
        ##########
        ########## New clustering test
        ##########
        # a = time.time()
        edge_index = coalesce(edge_index)
        cluster_mems, cluster_ids, cluster_batch, peaks, troughs = self.clusterer(elevation.unsqueeze(1), edge_index, batch)

        # Generate scaled pools
        pooled_x, cluster_batch, scaling = self.scale(x, elevation, cluster_mems, cluster_ids, cluster_batch)

        # Second LayerNorm
        # x = self.n2(pooled_x, cluster_batch)

        # Connect pools
        if self.gen_edges:
            edge_index = self.connect(cluster_mems, cluster_ids, batch, edge_index, peaks, troughs)

        x = pooled_x
        batch = cluster_batch
        perms = (cluster_mems, cluster_ids, scaling)

        # b = time.time()

        ##########
        ##########
        ##########
        '''
        c = time.time()
        cluster_info = self.get_clusters(x.to(cpu), 
                                         edge_index.to(cpu), 
                                         elevation.to(cpu), 
                                         batch.to(cpu))
        
        x, edge_index, batch, perms, elevation, scaling = cluster_info
        d = time.time()
        print("Old: {}".format(d - c))
        print("New: {}".format(b - a))
        '''

        # Not implemented yet
        edge_attr = None

        return x.to(cuda), edge_index.to(cuda), edge_attr, batch.to(cuda), (perms, scaling), elevation


    def get_elevations(self, x, batch) -> Tensor:

        # x_glob_avg = torch.index_select(gap(x, batch), 0, batch)
        # x_glob_max = torch.index_select(gmp(x, batch), 0, batch)
        # x_glob = x_glob_avg + x_glob_max
        # x = torch.cat((x, x_glob_avg, x_glob_max), dim=1)

        # Get the elevation and add small noise to break ties
        elevation = self.mapper(x)  # .to(cpu)

        # Push the peaks away from the troughs
        '''
        if self.training:
            means = torch.full(elevation.shape, 0.0, device=device)
            stds = scatter_std(elevation.squeeze(1).detach(), batch).to(device) / 10
            stds = torch.index_select(stds, 0, batch).unsqueeze(1)
            noise = torch.normal(means, stds)
            elevation = elevation + noise
        else:
            pass
        '''
        
        # elevation = torch.tanh(elevation)
        elevation = softmax(elevation, batch)
        
        # On the off chance the graph has one node and is fed with a batch size of one
        if elevation.shape.numel() == 1:
            return elevation
        else:
            return elevation.squeeze()

    def scale(self, x, elevation, cluster_mems, cluster_ids, cluster_batch):

        pooled_x        = scatter(src=x[cluster_mems], index=cluster_ids, dim=0, reduce='max')
        max_elevations  = scatter(src=elevation[cluster_mems], index=cluster_ids, dim=0, reduce='max')
        cluster_batch   = scatter(src=cluster_batch.float(), index=cluster_ids, dim=0, reduce='max').long()
        max_normed      = softmax(max_elevations, index=cluster_batch).unsqueeze(1)
        pooled_x        = pooled_x * max_normed

        return pooled_x, cluster_batch, max_normed

    def connect(self, cluster_mems, cluster_ids, batch, edge_index, peaks, troughs):


        ########### Adapted from ASAPool code, originally from DiffPool
        
        # Graph coarsening.
        N = self.num_nodes
        row, col = edge_index[0], edge_index[1]
        A = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, sparse_sizes=(N, N))

        C = peaks
        S = torch_sparse.index_select(S, 1, C)
        A = torch_sparse.matmul(torch_sparse.matmul(torch_sparse.t(S), A), S)

        row, col, _ = A.coo()
        edge_index = torch.stack([row, col], dim=0)
        
        ###########

        '''
        members, counts = cluster_mems.unique(return_counts=True)
        if counts[counts > 1].any():
            # Find contacts
            contacts        = torch.isin(cluster_mems, members[counts > 1])
            contact_nodes   = cluster_mems[contacts]
            contact_pools   = cluster_ids[contacts]
            contact_nodes, perm = contact_nodes.sort()
            contact_pools = contact_pools[perm]
            # Split, unify
            _, split_counts = contact_nodes.unique(return_counts=True)
            neighbors = torch.split(contact_pools, split_counts.tolist())
            new_edges = torch.cat([torch.combinations(i, with_replacement=True) for i in neighbors]).T
            # Implicit coalesce in to_undirected
            edge_index = to_undirected(new_edges)
            edge_index = add_self_loops(edge_index, num_nodes=cluster_ids.max() + 1)[0]
        else:
            # Only one cluster, so no contacts
            pass
        '''

        return edge_index
    
    def get_clusters(
        self, 
        x, 
        edge_index, 
        elevation, 
        batch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Cannot pool a single node
        if x.shape[0] == 1:
            return x, edge_index, batch, None, elevation, None

        pooled_nodes = torch.tensor([], dtype=torch.long)
        pooled_batch = torch.tensor([], dtype=torch.long)
        pooled_edges = edge_index.clone()
        pool_ids     = torch.tensor([], dtype=torch.long)

        filtered_elevation = elevation.clone()
        visited_nodes      = torch.full(elevation.shape, False)
        perms           = []
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
                                              perms, 
                                              clustered_nodes)
                
                total_pools, num_clusters, perms = batch_info
                
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
        
        scaled_x, pooled_edge_index, cluster_batches, promits = pooled_representation
        
        return scaled_x, pooled_edge_index, cluster_batches, perms, elevation, promits

    def agg_scale_connect(
        self, 
        x, 
        pooled_nodes, 
        pooled_edges, 
        pooled_batch, 
        elevation, 
        pool_ids
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Scale x by elevation
        # x = x * elevation.unsqueeze(1) / elevation.unsqueeze(1).detach()

        # Coalesce edge indices
        pooled_edge_index = coalesce(pooled_edges)

        # Generate clusters based on pool_ids
        pooled_x  = scatter(src=x[pooled_nodes], index=pool_ids, dim=0, reduce='max')

        # Scale pools
        # cluster_scales = self.scaler(pooled_x)
        # scaled_x = pooled_x * cluster_scales

        # Scale them by the softmax of their max elevations
        
        # min_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='min')
        max_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='max')
        # prominences     = max_elevations - min_elevations
        summits         = max_elevations
        promits         = summits  # prominences + summits
        cluster_batches = scatter(src=pooled_batch, index=pool_ids, dim=0, reduce='max')
        normed_promits  = softmax(promits, index=cluster_batches).unsqueeze(1)
        scaled_x        = pooled_x * normed_promits

        return scaled_x, pooled_edge_index, cluster_batches, normed_promits

    def reset_batch(self, 
                    total_pools, 
                    new_pool_ids,
                    num_clusters, 
                    perms, 
                    clustered_nodes
        ) -> Tuple[Tensor, Tensor, Tensor]:
        
        total_pools  = new_pool_ids.max() + 1
        num_clusters += 1

        # Top pool
        # if perms.numel() == 0:
        #     perms = clustered_nodes.clone()

        perms.append(clustered_nodes.clone())
        
        return total_pools, num_clusters, perms

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

        # Can't recurse any further if we're only given a single node without neighbors
        if (~just_neighbors).all():
            return node_indices, batch[node_indices]
        
        neighbor_indices = neighbors[just_neighbors]
        node_of_origin  = sources[just_neighbors]

        # Compare neighbor scores with source scores
        neighbor_scores = torch.round(elevation[neighbor_indices], decimals=4)
        source_scores   = torch.round(elevation[node_indices].index_select(0, node_of_origin), decimals=4)
        lower_neighbors = neighbor_indices[neighbor_scores <= source_scores].unique()  # Try <= with rounding to third decimal place

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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple, Tensor]:

        self.num_nodes = x.shape[0]

        if batch is None:
            self.batch = edge_index.new_zeros(x.size(0))

        elevation    = self.get_elevations(x, batch)
        cluster_info = self.get_clusters(x.to(cpu), 
                                         edge_index.to(cpu), 
                                         elevation.to(cpu), 
                                         batch.to(cpu))
        
        x, edge_index, batch, perms, elevation, promits = cluster_info

        # Not implemented yet
        edge_attr = None

        return x.to(cuda), edge_index.to(cuda), edge_attr, batch.to(cuda), (perms, promits), elevation

    def get_clusters(
        self, 
        x, 
        edge_index, 
        elevation, 
        batch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        # Cannot pool a single node
        if x.shape[0] == 1:
            return x, edge_index, batch, None, elevation, None

        pooled_nodes = torch.tensor([], dtype=torch.long)
        pooled_batch = torch.tensor([], dtype=torch.long)
        pooled_edges = edge_index.clone()
        pool_ids     = torch.tensor([], dtype=torch.long)

        filtered_elevation = elevation.clone()
        visited_nodes      = torch.full(elevation.shape, False)
        perms           = []
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
                                              perms, 
                                              clustered_nodes)
                
                total_pools, num_clusters, perms = batch_info
                
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
        
        scaled_x, pooled_edge_index, cluster_batches, promits = pooled_representation
        
        return scaled_x, pooled_edge_index, cluster_batches, perms, elevation, promits

    def agg_scale_connect(
        self, 
        x, 
        pooled_nodes, 
        pooled_edges, 
        pooled_batch, 
        elevation, 
        pool_ids
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Scale x by elevation
        # x = x * elevation.unsqueeze(1) / elevation.unsqueeze(1).detach()

        # Coalesce edge indices
        pooled_edge_index = coalesce(pooled_edges)

        # Generate clusters based on pool_ids
        pooled_x  = scatter(src=x[pooled_nodes], index=pool_ids, dim=0, reduce='max')

        # Scale pools
        # cluster_scales = self.scaler(pooled_x)
        # scaled_x = pooled_x * cluster_scales

        # Scale them by the softmax of their max elevations
        
        # min_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='min')
        max_elevations  = scatter(src=elevation[pooled_nodes], index=pool_ids, dim=0, reduce='max')
        # prominences     = max_elevations - min_elevations
        summits         = max_elevations
        promits         = summits  # prominences + summits
        cluster_batches = scatter(src=pooled_batch, index=pool_ids, dim=0, reduce='max')
        normed_promits  = softmax(promits, index=cluster_batches).unsqueeze(1)
        scaled_x        = pooled_x * normed_promits

        return scaled_x, pooled_edge_index, cluster_batches, normed_promits

    def reset_batch(self, 
                    total_pools, 
                    new_pool_ids,
                    num_clusters, 
                    perms, 
                    clustered_nodes
        ) -> Tuple[Tensor, Tensor, Tensor]:
        
        total_pools  = new_pool_ids.max() + 1
        num_clusters += 1

        # Top pool
        # if perms.numel() == 0:
        #     perms = clustered_nodes.clone()

        perms.append(clustered_nodes.clone())
        
        return total_pools, num_clusters, perms

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

        # Can't recurse any further if we're only given a single node without neighbors
        if (~just_neighbors).all():
            return node_indices, batch[node_indices]
        
        neighbor_indices = neighbors[just_neighbors]
        node_of_origin  = sources[just_neighbors]

        # Compare neighbor scores with source scores
        neighbor_scores = torch.round(elevation[neighbor_indices], decimals=4)
        source_scores   = torch.round(elevation[node_indices].index_select(0, node_of_origin), decimals=4)
        lower_neighbors = neighbor_indices[neighbor_scores <= source_scores].unique()  # Try <= with rounding to third decimal place

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

        x = self.n1(x)

        # Get the elevation and add small noise to break ties
        elevation = self.mapper(x).to(cpu)

        # if training:
        #     noise = torch.normal(torch.full(elevation.shape, 0.0), torch.full(elevation.shape, 0.1))
        #     elevation = elevation + noise
        
        elevation = softmax(elevation, batch.to(cpu))
        
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
