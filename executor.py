
import utils.models as models
import utils.plotting as plotting
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import torch
import torch.nn as nn
from tdc import Evaluator
import pandas as pd
import numpy as np
torch.set_default_dtype(torch.float32)
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

class build_model():
    def __init__(self, args, dataset):
        super().__init__()

        train_data, valid_data, test_data = dataset

        in_features = train_data[0].x.shape[1]
        num_classes = train_data[0].y.shape[0]  # Same for classification or regression because classifications are all binary

        if args.model_name == 'GCN':
            self.model = models.GCN(in_features, num_classes)
        elif args.model_name == 'TopKMulti':
            self.model = models.TopKGCNMultiTask(in_features, num_classes, args.min_val, args.task)
        elif args.model_name == 'TopoMulti':
            self.model = models.TopoGCNMultiTask(in_features, num_classes, args.min_val, args.task)
        elif args.model_name == 'GCNContourPool':
            self.model = models.GCNContourPool(in_features, num_classes, args.task)
        elif args.model_name == 'pADME':
            self.model = models.pADME(in_features, num_classes, args.task)
        elif args.model_name == 'Net':
            self.model = models.Net(in_features, num_classes, args.task)
        elif args.model_name == 'GCN_Pool':
            self.model = models.GCN_Pool(in_features, num_classes)
        elif args.model_name == 'TopPool':
            self.model = models.TopPool(in_features, num_classes)
        elif args.model_name == 'Benchmarker':
            self.model = models.BenchGNN(in_features, 
                                         hid_features=32, 
                                         out_features=num_classes,
                                         gnn_type=args.gnn_type,
                                         pool_type=args.pool_type)
        
        else:
            raise NotImplementedError

        self.model = self.model.to(cuda)

        # Create batched dataloader
        local_batch_size = args.batch_size
        self.train_loader = DataLoader(train_data, batch_size=local_batch_size, shuffle=False)
        self.valid_loader = DataLoader(valid_data, batch_size=local_batch_size, shuffle=False)
        self.test_loader = DataLoader(test_data, batch_size=1)
        self.batch_size = args.batch_size

        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # Loss function
        if args.task == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        elif args.task == 'regression':
            # self.loss_fn = nn.L1Loss()
            self.loss_fn = nn.MSELoss()

        # Epochs
        self.epochs = args.epochs

        # Trackers
        self.loss_track = []
        self.acc_track = []

        # Store valid performance over validation set - reset after each run
        self.valid_preds = pd.DataFrame()
        self.valid_label = pd.DataFrame()

        self.valid_perf = pd.DataFrame()  # Track valid performance while training
        self.test_perf = pd.DataFrame()

        self.args = args

        if args.task == 'classification':
            self.evaluator = Evaluator(name = 'ROC-AUC')
        elif args.task == 'regression':
            self.evaluator = Evaluator(name = 'MAE')
        else:
            raise Exception("--evaluator must be among the allowed")

    def train(self):

        # Run in training mode
        self.model.train()

        # Simplest training loop possible
        for e in tqdm(range(self.epochs)):

            loss_track = torch.tensor([0.0])
            # print("\nBeginning epoch {}/{}".format(e + 1, self.epochs))
            for batch in self.train_loader:

                self.optimizer.zero_grad()

                out = self.model(batch)
                targets = batch.y.to(torch.float32)

                # Detach unused tensors
                # out, targets = self.mask_output(out, targets)

                loss = self.loss_fn(out, targets)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                '''
                # Just nice to have on hand to check if the weights are updating
                names = [i for i in self.model.named_parameters()]
                print("\n\nMin/Max Gradients:")
                for i in range(len(names)):
                    print("  {}: {}, {}".format(names[i][0], round(names[i][1].grad.min().item(), 3), round(names[i][1].grad.max().item(), 3)))

                a = [i.clone() for i in list(self.model.parameters())]

                self.optimizer.step()

                b = [i.clone() for i in list(self.model.parameters())]
                updating = [not (torch.equal(i, j)) for i, j in list(zip(a, b))]
                names = [i[0] for i in self.model.named_parameters()]
                print("\n\nParameters updating:")
                for i in range(len(updating)):
                    print("  {}: {}".format(updating[i], names[i]))
                '''
                self.tracker(loss)
            
            # Run validation
            if e % 10 == 0:
                self.validate()
                # plotting.plot_loss(self.loss_track, epoch=e)
                # pass
        
        self.validate()

    # Simple validation run
    def validate(self):

        # Simplest validation loop possible
        with torch.no_grad():

            for batch_num, batch in enumerate(self.valid_loader):
                
                out = self.model(batch)
                targets = batch.y.to(torch.float32)

                # Update trackers
                self.track_valid_perf(out, targets)
                # self.model.explain_yourself(batch_num)

        
        self.report_valid(batch.target_names[0])

    
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

            # Only look at values for this particular endpoint
            y_pred = y_pred[y_true != -1]
            y_true = y_true[y_true != -1]

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

        # Save plot
        plot = self.valid_perf.plot()
        lgd = plot.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
        fig = plot.get_figure()
        fig.savefig("utils/plotting/valid_perf.png", bbox_extra_artists=[lgd], bbox_inches='tight')

    # TODO: update other tracking metrics, ideally use TensorBoard
    def tracker(self, loss):
        self.loss_track.append(loss.item())

    # Broadcast the outputs for the available tasks into a reduced tensor y_i \in R^t
    # This masks the gradients from the unlabeled tasks
    def mask_output(self, out, targets) -> tuple:
        out = out[targets != -1]
        targets = targets[targets != -1]
        return out, targets
    
    # This is adapted directly from their official testing code at:
    # https://tdcommons.ai/benchmark/overview/
    def test(self):

        print("\nBeginning test...\n")

        # Just use ours for now
        # test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        
        y_label = []
        y_pred = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader)):
                out = self.model(batch)
                y_label.extend(batch.y.flatten().tolist())
                y_pred.extend(out.flatten().tolist())
                self.model.explain_yourself(i)
                
        
        from sklearn.metrics import r2_score

        coefficient_of_dermination = r2_score(y_label, y_pred)

        print("\n")
        print("#"*40)
        print("#"*40)
        print("Final Test R2 Score: {}".format(coefficient_of_dermination))
        print("#"*40)
        print("#"*40)
        print("\n")

        return y_pred

    def save(self):
        raise NotImplementedError


# Just nice to have on hand to check if the weights are updating
# a = [i.clone() for i in list(self.model.parameters())]

# self.optimizer.step()

# b = [i.clone() for i in list(self.model.parameters())]
# updating = [not (torch.equal(i, j)) for i, j in list(zip(a, b))]
# names = [i[0] for i in self.model.named_parameters()]
# print("Parameters updating:")
# for i in range(len(updating)):
#     print("  {}: {}".format(updating[i], names[i]))
