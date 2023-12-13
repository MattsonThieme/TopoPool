from tdc.benchmark_group import admet_group
from utils import preprocess, executor
import numpy as np

def tu_test(args, gnn_type, pool_type):

    args.gnn_type = gnn_type
    args.pool_type = pool_type
    args.task = 'classification'

    perfs = []

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        
        print("#"*40)
        print("\nBeginning test run {}/10...\n".format(seed))
        print("#"*40)
  
        dataset = preprocess.load_data(args)
        model = executor.build_model(args, dataset)
        
        # Initiate training and testing
        model.train()
        y_pred_test = model.test_tu()

        perfs.append(y_pred_test)
    
    mean, std = np.array(perfs).mean(), np.array(perfs).std()

    return round(mean, 3), round(std, 3)


def tdc_test(args, gnn_type, pool_type):

    args.gnn_type  = gnn_type
    args.pool_type = pool_type

    group            = admet_group(path = 'data/')
    predictions_list = []
    perf_list        = []

    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print("\n")
        print("#"*40)
        print("#"*40)
        print("\nBeginning test run {}/10...\n".format(seed))
        print("#"*40)
        print("#"*40)
        print("\n")

        benchmark       = group.get(args.group_name) 
        predictions     = {}
        name            = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid    = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
        
        dataset = preprocess.load_data(args, train=train, valid=valid, test=test)
        model   = executor.build_model(args, dataset)

        # Initiate training and testing
        model.train()
        y_pred_test, y_label_test, perf = model.test()

        predictions[name] = y_pred_test
        predictions_list.append(predictions)
        perf_list.append(perf)
    
    avg = np.average(perf_list)
    std = np.std(perf_list)
    print("\n\n\n\nAvg: {}, Std: {}\n\n\n".format(avg, std))
    return {args.group_name: [avg, std]}
