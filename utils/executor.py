
import utils.models as models
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from tdc import Evaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float32)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


# Create a new benchmarker from scratch
class build_model():
    def __init__(self, args, dataset):
        super().__init__()

        self.args = args
        train_data, valid_data, test_data = dataset

        self.num_valid_examples = len(valid_data)
        self.num_test_examples  = len(test_data)

        in_features = train_data[0].x.shape[1]

        if args.task == 'classification':
            try:
                num_classes = train_data.num_classes
            # Binary classification for TDC datasets
            except:
                num_classes = 2 
        else:
            num_classes = 1

        self.model = models.BenchGNN(in_features, 
                                     hid_features=32, 
                                     out_features=num_classes,
                                     gnn_type=args.gnn_type,
                                     pool_type=args.pool_type,
                                     task=args.task)
        self.model = self.model.to(cuda)

        # Create batched dataloader
        local_batch_size  = args.batch_size
        self.train_loader = DataLoader(train_data, batch_size=local_batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_data, batch_size=local_batch_size, shuffle=True)
        self.test_loader  = DataLoader(test_data,  batch_size=1)
        self.batch_size   = args.batch_size

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)  #, weight_decay=0.00001)

        # Loss function
        if args.evaluator == 'MAE':
            self.loss_fn = nn.MSELoss()
            self.evaluator = Evaluator(name = 'MAE')
        elif args.evaluator == 'CrossEntropy':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception("Add entry in evaluators table")

        # Epochs
        self.epochs = args.epochs

        # Trackers
        self.loss_track = []
        self.acc_track  = []
        self.best_perf  = 0
        self.patience   = 0

        # Store valid performance over validation set - reset after each run
        self.valid_preds = pd.DataFrame()
        self.valid_label = pd.DataFrame()

        self.valid_cluster_size = []
        self.valid_perf         = pd.DataFrame()  # Track valid performance while training

    def load_weights(self, model_pth):
        self.model.load_state_dict(torch.load(model_pth))
    
    def eval_smiles(self, data):
        self.model(data)
        self.model.explain_yourself('custom')

    def train(self):

        best_perf = np.inf

        # Simplest training loop possible
        for e in tqdm(range(self.epochs)):
            for i, batch in enumerate(self.train_loader):
                # print("Batch number: {}".format(i))

                # Run in training mode
                self.model.train()

                self.optimizer.zero_grad()

                out = self.model(batch.to(cuda))
                targets = batch.y
                if self.args.task == 'regression':
                    targets = targets.to(torch.float32)
                else:
                    targets = targets.long()
                    if len(targets.shape) > 1:
                        targets = targets.squeeze(1)

                loss = self.loss_fn(out, targets)

                loss.backward()
                self.optimizer.step()
                self.tracker(loss)

            # Run validation, update patiences
            val_interval = 1
            if e % val_interval == 0 and e != 0:
                perf = self.validate()

                if perf < best_perf:
                    best_perf = perf
                    self.best_perf = perf
                    patience = 0
                    self.save_model()
                else:
                    patience += val_interval
                
                self.patience = patience
                
                if patience > self.args.patience:
                    print("Breaking out of loop at {} epochs with patience {}".format(e, self.args.patience))
                    break
        
        self.validate()

    def save_model(self, r2=None):
        if r2:
            torch.save(self.model.state_dict(), "saved_models/{}_{}_{}_TestR2-{}.pt".format(self.args.gnn_type,
                                                                                     self.args.pool_type,
                                                                                     self.args.group_name,
                                                                                     round(r2, 4)))
        else:
            torch.save(self.model.state_dict(), "saved_models/{}_{}_{}_bestValid.pt".format(self.args.gnn_type,
                                                                               self.args.pool_type,
                                                                               self.args.group_name))
    
    def load_best_valid(self):
        self.model.load_state_dict(torch.load(
            "saved_models/{}_{}_{}_bestValid.pt".format(self.args.gnn_type,
                                                     self.args.pool_type,
                                                     self.args.group_name)))

    # Simple validation run
    def validate(self):

        self.model.eval()

        # Simplest validation loop possible
        with torch.no_grad():

            if self.args.task == 'classification':

                losses = []
                for batch_num, batch in enumerate(self.valid_loader):
                    batch = batch.to(cuda)
                    pred = self.model(batch)

                    if len(batch.y.shape) > 1:
                        batch.y = batch.y.squeeze(1).long()
                    
                    losses.append(self.loss_fn(pred, batch.y).item())

                    if self.args.pool_type == 'Topo':
                        # num_clusters = self.model.explanations[5][0].shape[0]
                        mean_cluster_size = self.model.explanations[5][4][0][1].unique(return_counts=True)[1].float().mean()
                        self.valid_cluster_size.append(mean_cluster_size.to('cpu'))
                    else:
                        self.valid_cluster_size.append(0)
                
                perf = np.array(losses).mean()

                if self.args.pool_type == 'Topo':
                    # Uncomment to view the average cluster size over time
                    # Note: it slows down training
                    # plt.plot(self.valid_cluster_size)
                    # plt.title("Val Cluster Size | Best perf: {} | Patience: {}".format(round(self.best_perf, 3), self.patience))
                    # plt.savefig("utils/plotting/cluster_size.png")
                    # plt.close()
                    pass

                return perf

            else:
                
                losses = []
                for batch_num, batch in enumerate(self.valid_loader):
                    
                    out = self.model(batch)
                    targets = batch.y.to(torch.float32)
                    losses.append(self.loss_fn(out, targets).item())

                    # Update trackers
                    self.track_valid_perf(out, targets)
                    
                    if self.args.pool_type == 'Topo':
                        mean_cluster_size = self.model.explanations[5][4][0][1].unique(return_counts=True)[1].float().mean()
                        self.valid_cluster_size.append(mean_cluster_size.to('cpu'))
                        # plt.plot(self.valid_cluster_size)
                        # plt.title("Val Cluster Size | Best perf: {} | Patience: {}".format(round(self.best_perf, 3), self.patience))
                        # plt.savefig("utils/plotting/cluster_size.png")
                        # plt.close()
                    else:
                        self.valid_cluster_size.append(0)

        return np.array(losses).mean()

    # Append to the validation set trackers
    def track_valid_perf(self, preds, targets):
        self.valid_preds = self.valid_preds.append(pd.DataFrame(preds.tolist()))
        self.valid_label = self.valid_label.append(pd.DataFrame(targets.tolist()))
        
    # Summarize and record the validation performance per endpoint
    def report_valid(self, target_names):
        self.valid_preds.columns = target_names
        self.valid_label.columns = target_names
        
        perf_by_endpoint = []
        for ep in target_names:
            y_true = self.valid_label[ep].astype(float).to_numpy()
            y_pred = self.valid_preds[ep].astype(float).to_numpy()

            try:
                perf_by_endpoint.append(self.evaluator(y_true, y_pred))
            except:
                perf_by_endpoint.append(-1)
                print("\nNot enough data for {}".format(ep))
                print("If this is DEV MODE, just ignore this.")
                print("If it is not DEV MODE, there is a problem...\n")

        # Update global validation tracker
        scores = pd.DataFrame(perf_by_endpoint).T
        scores.columns = target_names
        self.valid_perf = self.valid_perf.append(scores)
        self.valid_perf = self.valid_perf.reset_index().drop(columns=['index'])

        # Reset trackers for the next run
        self.valid_preds = pd.DataFrame()
        self.valid_label = pd.DataFrame()
        
        return perf_by_endpoint[-1]

    # TODO: update other tracking metrics, ideally use TensorBoard
    def tracker(self, loss):
        self.loss_track.append(loss.item())

    # Test function for the TUDatasets
    def test_tu(self):

        print("\nBeginning test...\n")

        # Load the best model
        self.load_best_valid()

        with torch.no_grad():
            self.model.eval()
        
            correct = 0
            for batch_num, batch in tqdm(enumerate(self.test_loader)):
                batch = batch.to(cuda)
                pred = self.model(batch).max(dim=1)[1]
                if len(batch.y.shape) > 1:
                    batch.y = batch.y.squeeze(1)
                correct += pred.eq(batch.y).sum().item()
            
            perf = correct / self.num_test_examples
            
            try:
                self.model.explain_yourself(batch_num)
            except:
                # Hierarchical pooling model doesn't implement this now
                pass
        
        print("\n\nTest accuracy: {}\n\n".format(round(perf, 3)))

        self.save_model(r2=round(perf, 3))
        return perf
    
    # Test function for the TDC datasets
    def test(self):
        # This is adapted directly from the TDC official testing code at:
        # https://tdcommons.ai/benchmark/overview/

        print("\nBeginning test...\n")

        # Load the best model
        self.load_best_valid()

        y_label = []
        y_pred = []

        with torch.no_grad():
            self.model.eval()
            maes    = []
            correct = 0

            for i, batch in tqdm(enumerate(self.test_loader)):
                out = self.model(batch)

                if self.args.task == 'regression':
                    y_label.extend(batch.y.flatten().tolist())
                    y_pred.extend(out.flatten().tolist())

                    maes.append(self.evaluator(y_label, y_pred))

                else:
                    pred = out.max(dim=1)[1]
                    correct += pred.eq(batch.y).sum().item()
                    y_pred.extend(pred.flatten().tolist())
                    y_label.extend(batch.y.flatten().tolist())
                
                try:
                    self.model.explain_yourself(i)
                except:
                    # Hierarchical pooling model doesn't implement this now
                    pass
            
            if self.args.task == 'regression':
                mae = np.array(maes).mean()
                perf = mae
            else:
                acc = correct / self.num_test_examples
                perf = acc
        
        from sklearn.metrics import r2_score

        if self.args.evaluator == 'MAE':
            
            r2 = r2_score(y_label, y_pred)

            print("\n")
            print("#"*40)
            print("#"*40)
            print("Final Test R2 Score: {}".format(r2))
            print("#"*40)
            print("#"*40)
            print("\n")

        else:

            print("\n")
            print("#"*40)
            print("#"*40)
            print("Final Test Accuracy: {}".format(acc))
            print("#"*40)
            print("#"*40)
            print("\n")    

        self.save_model(r2=r2)

        try:
            y_pred = y_pred.cpu()
        except:
            pass

        return y_pred, y_label, perf

    def save(self):
        raise NotImplementedError

