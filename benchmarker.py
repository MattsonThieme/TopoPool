# Benchmarking script
# Use: python benchmarker.py

from utils import testing
import pandas as pd
import argparse
import config

parser = argparse.ArgumentParser()
parser.add_argument('--group_name', type=str, default='all',
                    help='Caco2_Wang | Solubility_AqSolDB | (Forthcoming: others)')
parser.add_argument('--pool_layer', type=str, default='all',
                    help='Pooling Layer type')
parser.add_argument('--gnn_type', type=str, default='GCN',
                    help='GNN type to benchmark. Use "all" to test on GCN, GAT and SAGE')
parser.add_argument('--prop_source', type=str, default='ADME',
                    help='Tox | ADME')
parser.add_argument('--epochs', type=float, default=10000,
                    help='Number of epochs to train')
args = parser.parse_args()

# Benchmark models to run
if args.group_name == 'all':
    group_names = ['ENZYMES', 'PROTEINS', 'MUTAG', 'Caco2_Wang', 'PPBR_AZ']
else:
    group_names = [args.group_name]

if args.gnn_type == 'all':
    gnn_types = ['GCN', 'GAT', 'SAGE']
else:
    gnn_types = [args.gnn_type]

if args.pool_layer == 'all':
    pool_types = ['Topo', 'None', 'TopK', 'SAG', 'Diff', 'Edge', 'ASA']
else:
    pool_types = [args.pool_layer]

summary             = pd.DataFrame()
summary['Base GNN'] = gnn_types
perf_tracker        = {}

print("\nRunning benchmarks on:")
print("   Datasets: {}".format(group_names))
print("   GNNs:     {}".format(gnn_types))
print("   Pools:    {}\n".format(pool_types))

for group_name in group_names:

    args.group_name = group_name
    args.lr         = config.lrs[group_name]
    args.patience   = config.patiences[group_name]
    args.evaluator  = config.evaluators[args.group_name][0]
    args.direction  = config.evaluators[args.group_name][1]
    args.batch_size = config.bs[args.group_name]

    for pool_type in pool_types:

        gnn_results = []

        for gnn_type in gnn_types:

            print("\n\n\nBeginning test run on {}-{}-{}...\n\n\n".format(args.group_name, gnn_type, pool_type))

            # TU Datasets
            if group_name in ['PROTEINS', 'ENZYMES', 'MUTAG', 'ZINC']:
                
                mean, std = testing.tu_test(args=args, gnn_type=gnn_type, pool_type=pool_type)

                print("\n")
                print("#"*40)
                print("#"*40)
                print("Final results for {}-{}:".format(gnn_type, pool_type))
                print("    {}".format({args.group_name: [mean, std]}))
                print("#"*40)
                print("#"*40)
                print("\n")
            
            # TDC Datasets
            elif group_name in ['Caco2_Wang', 'PPBR_AZ', 'CYP2C19_Veith']:

                results = testing.tdc_test(args=args, gnn_type=gnn_type, pool_type=pool_type)
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
            
            else:
                raise Exception("args.group_name not recognized, use one of: 'PROTEINS', 'ENZYMES', 'MUTAG', 'Caco2_Wang', 'PPBR_AZ'")
            
            gnn_results.append((mean, std))

        summary[pool_type] = gnn_results
        summary.to_csv('results/{}_results.csv'.format(args.group_name))

    summary.to_csv('results/{}_results.csv'.format(args.group_name))
