import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GATConv, BatchNorm, SAGEConv, GATv2Conv, dense_diff_pool
from torch_geometric.nn.pool import TopKPooling, SAGPooling, EdgePooling
from torch_geometric.nn.dense import diff_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, to_dense_batch
from utils.topo_pool import TopoPool
from torch_geometric.data import Data
import torch.nn as nn
import pickle
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


# Benchmarking GNN model
class BenchGNN(torch.nn.Module):
    def __init__(
        self, 
        in_features:  int, 
        hid_features: int,
        out_features: int,
        gnn_type:     str,
        pool_type:    str):
        
        super(BenchGNN, self).__init__()
        
        self.gnn_type      = gnn_type
        self.pool_type     = pool_type
        self.num_features  = in_features
        self.nhid          = hid_features
        self.num_classes   = out_features
        
        # Define GNN layers
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(self.num_features * 2, self.nhid)
            self.conv2 = GCNConv(self.nhid * 2, self.nhid)
            self.conv3 = GCNConv(self.nhid * 2, self.nhid)
        
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(self.num_features * 2, self.nhid)
            self.conv2 = GATConv(self.nhid * 2, self.nhid)
            self.conv3 = GATConv(self.nhid * 2, self.nhid)
        
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels=self.num_features * 2, out_channels=self.nhid)
            self.conv2 = SAGEConv(in_channels=self.nhid * 2, out_channels=self.nhid)
            self.conv3 = SAGEConv(in_channels=self.nhid * 2, out_channels=self.nhid)

        else:
            raise Exception("gnn_layer must be in ['None', 'GCN', 'GAT', 'SAGE']")
        
        # Define pooling layer
        if pool_type == 'None':
            self.pool = False
        
        else:
            self.pool = True
    
            if pool_type == 'Topo':
                self.pool_layer = TopoPool(in_channels=self.nhid, max_clusters=None)

            elif pool_type == 'TopK':
                self.pool_layer = TopKPooling(in_channels=self.nhid)
            
            elif pool_type == 'SAG':
                self.pool_layer = SAGPooling(in_channels=self.nhid)
            
            elif pool_type == 'Edge':
                self.pool_layer = EdgePooling(in_channels=self.nhid)

            elif pool_type == 'Diff':
                num_nodes = 10
                self.gnn1_pool = GCNConv(self.nhid, self.nhid)
                self.gnn2_pool = GCNConv(self.nhid, num_nodes)

        # Final linear layers
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        mols, target_names, targets = data.mols, data.target_names, data.y

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = F.relu(self.conv2(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = self.conv3(x, edge_index)
        # x = F.dropout(x, p=0.5, training=self.training)

        if self.pool:
            if self.pool_type == 'Edge':
                pooling = self.pool_layer(x, edge_index=edge_index, batch=batch)
                x, edge_index, batch, _ = pooling

            elif self.pool_type == 'Diff':
                s = F.relu(self.gnn1_pool(x, edge_index))
                s = self.gnn2_pool(s, edge_index)
                
                adj          = to_dense_adj(edge_index, batch)
                s, _         = to_dense_batch(s, batch)
                x, mask      = to_dense_batch(x, batch)
                x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
                pooling      = None  # dense_diff_pool does not return assignments

                batch = torch.range(0, batch.max()).unsqueeze(1).long()
                batch = batch.expand(batch.shape[0], x.shape[1])
                batch = batch.flatten().to(cuda)

                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                

            else:
                pooling = self.pool_layer(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _, _ = pooling
                
        else:
            pooling = None

        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        # Track this pooling layer
        self.explanations = x, batch, mols, targets, target_names, pooling

        return x

    # Calls the visualizer on the last batch    
    def explain_yourself(self, batch_num):
        
        # visualization.vis_nodes(self.explanations)
        with open('explanations/{}_{}-{}.pkl'.format(batch_num,
                                                          self.gnn_type, 
                                                          self.pool_type),
                                                          "wb") as file:
            pickle.dump(self.explanations, file)


# Adapted from SAG Pooling paper
class GCN_Pool(torch.nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        task='classification'):
        
        super(GCN_Pool, self).__init__()
        self.num_features = in_features
        self.nhid = 32
        self.num_classes = 1
        # self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = 0.5
        
        self.conv1 = GCNConv(self.num_features * 2, self.nhid)
        # self.conv1 = GATConv(self.num_features, self.nhid, dropout=0.5)
        # self.conv1 = SAGEConv(in_channels=self.num_features, out_channels=self.nhid)
        # self.pool1 = ContourPoolingV3(in_channels=self.nhid, max_clusters=None)
        self.conv2 = GCNConv(self.nhid * 2, self.nhid)
        # self.conv2 = GATConv(self.nhid, self.nhid, dropout=0.5)
        # self.conv2 = SAGEConv(in_channels=self.nhid, out_channels=self.nhid)
        # self.pool2 = ContourPoolingV3(in_channels=self.nhid, max_clusters=None)
        self.conv3 = GCNConv(self.nhid * 2, self.nhid)
        # self.conv3 = GATConv(self.nhid, self.nhid, dropout=0.5)
        # self.conv3 = SAGEConv(in_channels=self.nhid, out_channels=self.nhid)
        self.pool3 = TopoPool(in_channels=self.nhid, max_clusters=None)
        # self.pool3 = TopKPooling(in_channels=self.nhid)
        # self.pool3 = SAGPooling(in_channels=self.nhid)
        # self.pool3 = EdgePooling(in_channels=self.nhid)  # Output signature contains 4, not 6 elements

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, mols, target_names, targets = data.x, data.edge_index, data.batch, data.mols, data.target_names, data.y

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = F.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = self.conv3(x, edge_index)

        pooling = self.pool3(x, edge_index=edge_index, batch=batch)
        x, edge_index, _, batch, _, _ = pooling
        # pooling = None
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)
        x = self.lin3(x)

        # Track this pooling layer
        self.explanations = x, batch, mols, targets, target_names, pooling

        return x

    # Calls the visualizer on the last batch    
    def explain_yourself(self, batch_num):
        
        # visualization.vis_nodes(self.explanations)
        with open('explanations/{}_GCN-Topo.pkl'.format(batch_num), "wb") as file:
            pickle.dump(self.explanations, file)

# Multi-task learner using GCN + TopK Sampling with min_value = 0.5
class TopKGCNMultiTask(torch.nn.Module):
    def __init__(self, in_features, out_features, min_val=0.5, model_task='classification') -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        ####################################
        ######## Shared GNN Layers #########
        ####################################

        out_global = 128
        self.gconv1 = GCNConv(in_features, out_global)
        self.gconv2 = GCNConv(out_global, out_global)
        self.drop = nn.Dropout(0.5)

        ####################################
        ###### Individual task heads #######
        ####################################

        # Task Pooling 1
        self.tpool1 = nn.ModuleList([MyTopKPooling(in_channels=out_global) for i in range(out_features)])

        # Task GNN
        out_t1 = 64
        self.tconv1 = nn.ModuleList([GCNConv(out_global, out_t1) for i in range(out_features)])

        # Final linear layer
        self.tlin1 = nn.ModuleList([Linear(out_t1, 1) for i in range(out_features)])

        # Final activation for classification tasks
        self.model_task = model_task
        self.final_act = nn.Sigmoid()

        ####################################
        # Variables for accessing explanations
        ####################################

        self.explanations = None
        
    def forward(self, data):

        x, edge_index, mols, target_names, targets = data.x, data.edge_index, data.mols, data.target_names, data.y

        ####################################
        ######### GLOBAL ACTIONS ###########
        ####################################

        x = self.gconv1(x, edge_index)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.gconv2(x, edge_index)
        x = torch.relu(x)
        x = self.drop(x)

        ####################################
        ########## TASK ACTIONS ############
        ####################################

        out = torch.tensor([])
        pooled_results = []
        batch = self.get_batch_labels(data.num_atoms)

        for i in range(self.out_features):

            # Pooling is organized as [pooled1, edge_index1, edge_attr1, batch1, perm1, score1] for each out_feature
            pooling = self.tpool1[i](x, edge_index, self.training, batch=batch)
            xt, local_edges, local_batches = pooling[0], pooling[1], pooling[3]

            xt = self.tconv1[i](xt, local_edges)

            # Aggregate pooled nodes per molecule
            xt = scatter(xt, dim=0, index=local_batches)

            # Final linear layer
            xt = self.tlin1[i](xt)

            # Final activation
            if self.model_task == 'classification':
                xt = self.final_act(xt)

            out = torch.cat((out, xt), dim=1)

            pooled_results.append(pooling)

        # Track this pooling layer
        self.explanations = torch.flatten(out), batch, mols, targets, target_names, pooled_results

        return torch.flatten(out).unsqueeze(1)
    
    # Get a mask assigning each node to a batch
    def get_batch_labels(self, num_atoms) -> torch.Tensor:
        batch = []
        for i, num_atoms in enumerate(num_atoms.tolist()):
            batch.extend([i] * num_atoms)
        
        return torch.tensor(batch)

    # Calls the visualizer on the last batch    
    def explain_yourself(self, batch_num):
        
        # visualization.vis_nodes(self.explanations)
        with open('data/processed/explanations_batch-{}.pkl'.format(batch_num), "wb") as file:
            pickle.dump(self.explanations, file)
