import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GCNConv, GATConv, LayerNorm, SAGEConv, dense_diff_pool
from torch_geometric.nn.pool import TopKPooling, SAGPooling, EdgePooling, ASAPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
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
        pool_type:    str,
        task:         str):
        
        super(BenchGNN, self).__init__()
        
        self.gnn_type      = gnn_type
        self.pool_type     = pool_type
        self.num_features  = in_features
        self.nhid          = hid_features
        self.num_classes   = out_features
        self.task          = task
        
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
                self.pool_layer = TopoPool(in_channels=self.nhid, max_clusters=None, gen_edges=False)

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
            
            elif pool_type == 'ASA':
                self.pool_layer = ASAPooling(in_channels=self.nhid)

        # Norm layers
        self.n1 = LayerNorm(self.nhid)
        self.n2 = LayerNorm(self.nhid)
        self.n3 = LayerNorm(self.nhid)
        self.n4 = LayerNorm(self.nhid)

        # Final linear layers
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        try:
            mols, target_names, targets = data.mols, data.target_names, data.y
        except:
            # TUDataset doesnt come with these
            mols, target_names, targets = None, None, None
        
        x_glob = torch.index_select(gap(x, batch), 0, batch)  # Simulates a global node
        x = torch.cat((x, x_glob), dim=1)
        x = self.conv1(x, edge_index)
        x = self.n1(x,  batch)
        x = F.leaky_relu(x)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = self.conv2(x, edge_index)
        x = self.n2(x,  batch)
        x = F.leaky_relu(x)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x_glob = torch.index_select(gap(x, batch), 0, batch)
        x = torch.cat((x, x_glob), dim=1)
        x = self.conv3(x, edge_index)
        x = self.n3(x,  batch)
        x = F.leaky_relu(x)
        
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
            
            elif self.pool_type == 'ASA':
                pooling = self.pool_layer(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _ = pooling
                
            else:
                pooling = self.pool_layer(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _, _ = pooling
            
            x = self.n3(x,  batch)
                
        else:
            pooling = None
        
        
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))

        if self.task == 'classification':
            x = F.log_softmax(self.lin3(x), dim=-1)
        else:
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


# Optional hierarchical model. Note: hyperparameters have not been tuned for this model
class BenchGNN_Hierarchical(torch.nn.Module):
    def __init__(
        self, 
        in_features:  int, 
        hid_features: int,
        out_features: int,
        gnn_type:     str,
        pool_type:    str,
        task:         str):
        
        super(BenchGNN_Hierarchical, self).__init__()
        
        self.gnn_type      = gnn_type
        self.pool_type     = pool_type
        self.num_features  = in_features
        self.nhid          = hid_features
        self.num_classes   = out_features
        self.task          = task
        
        # Define GNN layers
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(self.num_features, self.nhid)
            self.conv2 = GCNConv(self.nhid, self.nhid)
            self.conv3 = GCNConv(self.nhid, self.nhid)
            self.conv4 = GCNConv(self.nhid, self.nhid)
            self.conv5 = GCNConv(self.nhid, self.nhid)
        
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(self.num_features * 2, self.nhid)
            self.conv2 = GATConv(self.nhid * 2, self.nhid)
            self.conv3 = GATConv(self.nhid * 2, self.nhid)
            self.conv4 = GATConv(self.nhid * 2, self.nhid)
        
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels=self.num_features, out_channels=self.nhid)
            self.conv2 = SAGEConv(in_channels=self.nhid, out_channels=self.nhid)
            self.conv3 = SAGEConv(in_channels=self.nhid, out_channels=self.nhid)
            self.conv4 = SAGEConv(in_channels=self.nhid, out_channels=self.nhid)

        else:
            raise Exception("gnn_layer must be in ['None', 'GCN', 'GAT', 'SAGE']")
        
        # Define pooling layer
        if pool_type == 'None':
            self.pool = False
        
        else:
            self.pool = True
    
            if pool_type == 'Topo':
                self.pool_layer1 = TopoPool(in_channels=self.nhid, max_clusters=None, gen_edges=True)
                self.pool_layer2 = TopoPool(in_channels=self.nhid, max_clusters=None, gen_edges=True)

            elif pool_type == 'TopK':
                self.pool_layer1 = TopKPooling(in_channels=self.nhid)
                self.pool_layer2 = TopKPooling(in_channels=self.nhid)
            
            elif pool_type == 'SAG':
                self.pool_layer1 = SAGPooling(in_channels=self.nhid)
                self.pool_layer2 = SAGPooling(in_channels=self.nhid)

            elif pool_type == 'Edge':
                self.pool_layer1 = EdgePooling(in_channels=self.nhid)
                self.pool_layer2 = EdgePooling(in_channels=self.nhid)

            elif pool_type == 'Diff':
                num_nodes1 = 10
                self.gnn1_pool = GCNConv(self.nhid, self.nhid)
                self.gnn2_pool = GCNConv(self.nhid, num_nodes1)

                num_nodes2 = 5
                self.gnn3_pool = GCNConv(self.nhid, self.nhid)
                self.gnn4_pool = GCNConv(self.nhid, num_nodes2)
            
            elif pool_type == 'ASA':
                self.pool_layer1 = ASAPooling(in_channels=self.nhid)
                self.pool_layer2 = ASAPooling(in_channels=self.nhid)

        # Norm layers
        self.n1 = LayerNorm(self.nhid)
        self.n2 = LayerNorm(self.nhid)
        self.n3 = LayerNorm(self.nhid)

        # Final linear layers
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        try:
            mols, target_names, targets = data.mols, data.target_names, data.y
        except:
            # TUDataset doesnt come with these
            mols, target_names, targets = None, None, None

        x = self.conv1(x, edge_index)
        x = self.n1(x,  batch)
        x = F.leaky_relu(x)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if self.pool:
            if self.pool_type == 'Edge':
                pooling1 = self.pool_layer1(x, edge_index=edge_index, batch=batch)
                x, edge_index, batch, _ = pooling1

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
                edge_index = dense_to_sparse(adj)[0]
                pooling1 = None
            
            elif self.pool_type == 'ASA':
                pooling1 = self.pool_layer1(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _ = pooling1

            else:
                pooling1 = self.pool_layer1(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _, _ = pooling1
                
        else:
            pooling1 = None

        x = self.conv2(x, edge_index)
        x = self.n2(x,  batch)
        x = F.leaky_relu(x)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if self.pool:
            if self.pool_type == 'Edge':
                pooling = self.pool_layer2(x, edge_index=edge_index, batch=batch)
                x, edge_index, batch, _ = pooling

            elif self.pool_type == 'Diff':
                s = F.relu(self.gnn3_pool(x, edge_index))
                s = self.gnn4_pool(s, edge_index)
                
                adj          = to_dense_adj(edge_index, batch)
                s, _         = to_dense_batch(s, batch)
                x, mask      = to_dense_batch(x, batch)
                x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
                pooling      = None  # dense_diff_pool does not return assignments

                batch = torch.range(0, batch.max()).unsqueeze(1).long()
                batch = batch.expand(batch.shape[0], x.shape[1])
                batch = batch.flatten().to(cuda)

                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                edge_index = dense_to_sparse(adj)[0]
                pooling2 = None
            
            elif self.pool_type == 'ASA':
                pooling2 = self.pool_layer2(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _ = pooling2

            else:
                pooling2 = self.pool_layer2(x, edge_index=edge_index, batch=batch)
                x, edge_index, _, batch, _, elevation2 = pooling2
                
        else:
            pooling2 = None

        x = self.conv3(x, edge_index)
        x = self.n3(x, batch)
        x = F.leaky_relu(x)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))

        if self.task == 'classification':
            x = F.log_softmax(self.lin3(x), dim=-1)
        else:
            x = self.lin3(x)

        # Track this pooling layer
        self.explanations = x, batch, mols, targets, target_names, pooling1

        return x

    # Calls the visualizer on the last batch    
    def explain_yourself(self, batch_num):
        
        # visualization.vis_nodes(self.explanations)
        with open('explanations/{}_{}-{}.pkl'.format(batch_num,
                                                          self.gnn_type, 
                                                          self.pool_type),
                                                          "wb") as file:
            pickle.dump(self.explanations, file)
