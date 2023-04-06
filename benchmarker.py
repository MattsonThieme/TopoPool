
# This implements the official test script from TDC
# https://tdcommons.ai/benchmark/overview/

import argparse
from utils import preprocess, executor
from tdc.benchmark_group import admet_group

parser = argparse.ArgumentParser()
parser.add_argument('--group_name', type=str, default='Caco2_Wang',
                    help='Caco2_Wang | Solubility_AqSolDB | (Forthcoming: others)')
parser.add_argument('--official_test', type=bool, default=True,
                    help='True for this script')
parser.add_argument('--model_name', type=str, default='Benchmarker',
                    help='GCN | TopKMulti | TopoMulti | GCNContourPool | pADME | Net | GCN_Pool | Benchmarker')
parser.add_argument('--task', type=str, default='regression',
                    help='classification | regression | all')
parser.add_argument('--data_source', type=str, default='TDC',
                    help='TDC | (Forthcoming: AbbVie Internal)')
parser.add_argument('--prop_source', type=str, default='ADME',
                    help='Tox | ADME')
parser.add_argument('--endpoints', type=list, default=['caco2_wang'], #'lipophilicity_astrazeneca', 'solubility_aqsoldb', 'caco2_wang'],
                    help='Specify particular endpoints within each source. Enter None to use all the endpoints for that task (classification / regression).')
parser.add_argument('--dev_mode', type=bool, default=False,
                    help='Development mode, everything is smaller (only 1k examples in train/valid/test sets)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Training batch size.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Learning rate.')
parser.add_argument('--epochs', type=float, default=100,
                    help='Number of epochs to train')
parser.add_argument('--patience', type=float, default=50,
                    help='Number of epochs to train')
parser.add_argument('--reprocess', type=bool, default=True,
                    help='Reprocess data from scratch')
parser.add_argument('--min-val', type=float, default=0.5,
                    help='Min val for top-k selection')
parser.add_argument('--gnn_layer', type=str,
                    help='GNN Layer type')
parser.add_argument('--pool_layer', type=str,
                    help='Pooling Layer type')
args = parser.parse_args()

def run_test(args, gnn_type, pool_type):

    args.gnn_type = gnn_type
    args.pool_type = pool_type

    group = admet_group(path = 'data/')
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]: #, 6, 7, 8, 9, 10]:
        print("\n")
        print("#"*40)
        print("#"*40)
        print("\nBeginning test run {}/5...\n".format(seed))
        print("#"*40)
        print("#"*40)
        print("\n")

        benchmark = group.get(args.group_name) 
        # all benchmark names in a benchmark group are stored in group.dataset_names
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        
            # --------------------------------------------- # 
            #  Train your model using train, valid, test    #
            #  Save test prediction in y_pred_test variable #
            # --------------------------------------------- #

        # Load and preprocess data exactly as given
        dataset = preprocess.load_data(args, train=train, valid=valid, test=test)
        
        # Instantiate model
        model = executor.build_model(args, dataset)

        # Initiate Training, Validation is baked in
        model.train()

        # Initiate Testing
        y_pred_test = model.test()

            # --------------------------------------------- # 
            #                End of our code                #
            # --------------------------------------------- #
            
        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    return group.evaluate_many(predictions_list)

# Benchmark models to run
gnn_types  = ['GCN']
pool_types = ['None', 'TopK', 'SAG', 'Edge', 'Diff', 'Topo']

import pandas as pd
summary = pd.DataFrame()
summary['Base GNN'] = gnn_types

perf_tracker = {}

for pool_type in pool_types:
    
    gnn_results = []

    for gnn_type in gnn_types:

        results = run_test(args=args, gnn_type=gnn_type, pool_type=pool_type)
        perf_tracker[gnn_type + ' ' + str(pool_type)] = results

        print("\n")
        print("#"*40)
        print("#"*40)
        print("Final results for {}-{}:".format(gnn_type, pool_type))
        print("    {}".format(results))
        print("#"*40)
        print("#"*40)
        print("\n")

        key = list(results.keys())[0]
        mean, std = results[key][0], results[key][1]
        gnn_results.append((mean, std))

    summary[pool_type] = gnn_results

summary.to_csv('results/{}_results.csv'.format(args.group_name))
