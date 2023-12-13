# Download and process TDC data

from tdc.single_pred import ADME, Tox
from tdc.chem_utils import MolConvert
from tdc.utils import retrieve_label_name_list, retrieve_dataset_names
from torch_geometric.utils import contains_isolated_nodes, remove_isolated_nodes, add_self_loops, remove_self_loops
from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from rdkit import Chem
import pickle
import os.path
import pandas as pd
import torch
cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')


# Load data
def load_data(args, train=None, valid=None, test=None) -> torch.utils.data.Dataset:

    if args.group_name in ['PROTEINS', 'ENZYMES', 'MUTAG']:
        path = './data/{}'.format(args.group_name)
        dataset = TUDataset(path, name=args.group_name).shuffle()
        n = (len(dataset) + 9) // 10
        test_data = dataset[:n]
        valid_data = dataset[n:2 * n]
        train_data = dataset[2 * n:]
        
        return train_data, valid_data, test_data


    dataset_to_use = 'DEV' if args.dev_mode else 'FULL'

    # If official_test, use the provided train/valid/test sets
    if args.official_test:
        train = train.drop(['Drug_ID'], axis=1)
        valid = valid.drop(['Drug_ID'], axis=1)
        test = test.drop(['Drug_ID'], axis=1)

        # train, valid, test = norm_sets(train, valid, test)

        train_data, valid_data, test_data = gen_pyg_dataset(args, train, valid, test)

        # TODO: HACKY WORKAROUND, FIX LATER
        # Half life values have a huge range, predict their log, then exponentiate
        if args.group_name in ['Half_Life_Obach', 'Clearance_Hepatocyte_AZ']:
            for i, ex in enumerate(train_data):
                train_data[i].y = torch.log(train_data[i].y)
            
            for i, ex in enumerate(valid_data):
                valid_data[i].y = torch.log(valid_data[i].y)
            
            for i, ex in enumerate(test_data):
                test_data[i].y = torch.log(test_data[i].y)

        
        return train_data, valid_data, test_data
    
    else:
        # Try loading saved data
        if os.path.isfile("data/processed/{}_{}_train_data_pyg_{}.pkl".format(args.data_source, args.prop_source, dataset_to_use)) and not args.reprocess:
            
            print("\nLoading processed {} data from {}...".format(args.prop_source, args.data_source))

            with open("data/processed/{}_{}_train_data_pyg_{}.pkl".format(args.data_source, args.prop_source, dataset_to_use), "rb") as file:
                train_data = pickle.load(file)
            with open("data/processed/{}_{}_valid_data_pyg_{}.pkl".format(args.data_source, args.prop_source, dataset_to_use), "rb") as file:
                valid_data = pickle.load(file)
            with open("data/processed/{}_{}_test_data_pyg_{}.pkl".format(args.data_source, args.prop_source, dataset_to_use), "rb") as file:
                test_data = pickle.load(file)
            
            return train_data, valid_data, test_data

    prop_source = args.prop_source

    # Grab the list of relevant groups
    groups = retrieve_dataset_names(prop_source)
    if prop_source == 'Tox':
        groups.remove('toxcast')  # 617 labels, too many for now
    elif prop_source == 'ADME':
        pass
    
    # Filter for pre-defined endpoints (if applicable)
    if args.endpoints:
        groups = [i for i in groups if i in args.endpoints]

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for group in groups:

        # Get all labels for this group
        try:
            label_list = retrieve_label_name_list(group)
        except:
            label_list = None
        
        # Fill train/valid/test dataframes for each label
        if label_list:
            for label in label_list:
                train_data, valid_data, test_data = format(args, group, label, train_data, valid_data, test_data)
        else:
            train_data, valid_data, test_data = format(args, group, label_list, train_data, valid_data, test_data)

    # Save the raw data
    save_csv(args, train_data, valid_data, test_data)

    # Transform into pyg objects for training
    train_data, valid_data, test_data = gen_pyg_dataset(args, train_data, valid_data, test_data)

    return train_data, valid_data, test_data

def norm_sets(train, valid, test):

    mean = train.Y.mean()
    std = train.Y.std()

    train.Y = (train.Y - mean) / std
    valid.Y = (valid.Y - mean) / std
    test.Y = (test.Y - mean) / std

    return train, valid, test

# Format the input data
def format(args, group, label, train_data, valid_data, test_data) -> tuple:

    # Some groups have multiple labels
    if label:
        if args.prop_source == 'Tox':
            data = Tox(name = group, label_name = label)
        elif args.prop_source == 'ADME':
            data = ADME(name = group, label_name = label)
        
        # For naming the target column
        label = group + "_" + label
    else:
        if args.prop_source == 'Tox':
            data = Tox(name = group)
        elif args.prop_source == 'ADME':
            data = ADME(name = group)
        
        # For naming the target column
        label = group

    split = data.get_split()
    train = split['train']
    valid = split['valid']
    test = split['test']

    # Filter for the desired task type (classification / regression)
    if correct_task(args, train):

        # Drop irrelevant columns
        train = train.rename(columns={'Y': label}).drop(['Drug_ID'], axis=1)
        valid = valid.rename(columns={'Y': label}).drop(['Drug_ID'], axis=1)
        test = test.rename(columns={'Y': label}).drop(['Drug_ID'], axis=1)

        # Merge the datasets
        if train_data.shape[0] == 0:
            train_data = train
            valid_data = valid
            test_data = test
        else:
            train_data = train_data.merge(train, on='Drug', how='outer')
            valid_data = valid_data.merge(valid, on='Drug', how='outer')
            test_data = test_data.merge(test, on='Drug', how='outer')
        
    return train_data, valid_data, test_data

# Save the dataframes to CSV
def save_csv(args, train_data, valid_data, test_data) -> None:
    dataset_to_use = 'DEV' if args.dev_mode else 'FULL'
    train_data.to_csv("data/processed/{}_train_{}.csv".format(args.prop_source, dataset_to_use))
    valid_data.to_csv("data/processed/{}_valid_{}.csv".format(args.prop_source, dataset_to_use))
    test_data.to_csv("data/processed/{}_test_{}.csv".format(args.prop_source, dataset_to_use))


# Filter by classification or regression tasks
def correct_task(args, train_data):

    targets = train_data.Y

    # TODO: Kindof hacky / error prone way to separate classification and regression, fix later
    if args.task == 'classification':
        return True if (targets.max() == 1) & (targets.min() == 0) else False
    elif args.task == 'regression':
        return True if not ((targets.max() == 1) & (targets.min() == 0)) else False
    elif args.task == 'all':
        return True
    else:
        raise Exception("--task must be either 'classification', 'regression', or 'all'")


def custom_smiles(smiles_string):

    # Converter from SMILES to pyg
    converter = MolConvert(src = 'SMILES', dst = 'PyG')

    data_pyg = converter([smiles_string])

    # Just for creating the model, not used
    data_pyg.y = torch.tensor([0.0])

    return data_pyg


# Save data to pyg format for training
def gen_pyg_dataset(args, train_data, valid_data, test_data) -> tuple:

    train_data = train_data.fillna(-1)
    valid_data = valid_data.fillna(-1)
    test_data  = test_data.fillna(-1)

    train_data = make_connected(train_data)
    valid_data = make_connected(valid_data)
    test_data  = make_connected(test_data)

    # Converter from SMILES to pyg
    converter = MolConvert(src = 'SMILES', dst = 'PyG')

    if args.dev_mode:
        num_examples = 1000
        train_data = train_data.head(num_examples)
        valid_data = valid_data.head(num_examples)
        test_data = test_data.head(num_examples)

    # print("\nProcessing and saving training set...")
    train_data_pyg = converter(train_data.Drug.tolist())
    train_data_pyg = assign_labels(args, train_data_pyg, train_data)
    save_pyg_dataset(args, train_data_pyg, 'train')

    # print("\nProcessing and saving validation set...")
    valid_data_pyg = converter(valid_data.Drug.tolist())
    valid_data_pyg = assign_labels(args, valid_data_pyg, valid_data)
    save_pyg_dataset(args, valid_data_pyg, 'valid')

    # print("\nProcessing and saving testing set...")
    test_data_pyg = converter(test_data.Drug.tolist())
    test_data_pyg = assign_labels(args, test_data_pyg, test_data)
    save_pyg_dataset(args, test_data_pyg, 'test')

    return train_data_pyg, valid_data_pyg, test_data_pyg


# Remove isolated nodes separated by . in smiles strings
def make_connected(data):
    connected = []
    for mol in data.Drug.tolist():
        if '.' in mol:
            new_mol = mol.split('.')
            lens = [len(i) for i in new_mol]
            actual = max(lens)
            corrected_mol = new_mol[lens.index(actual)]
            connected.append(corrected_mol)
        else:
            connected.append(mol)
    data = data.assign(Drug = connected)
    return data


# Get the label for a given molecule
def assign_labels(args, pyg_objects, data) -> list:

    mols = data['Drug'].tolist()

    labeled_data = []

    for i, mol in enumerate(mols):

        # Grab the labels for this molecule, slice off the molecule SMILES string
        example = data[data['Drug'] == mol]

        if args.official_test:
            
            targets = example.Y.values.astype(float)
            targets = torch.tensor(targets).unsqueeze(-1)
            # Sometimes there are multiple measurements for the same drug. In this case, use the mean value
            if targets.shape[0] != 1:
                # print("\n{} entries for {}, using mean value of {}".format(targets.shape[0], mol, round(targets.mean().item(), 2)))
                targets = targets.mean().unsqueeze(-1).unsqueeze(-1)
            
        # Only keep molecules with at least three edges
        else:  # pyg_objects[i].x.shape[0] > 1 and pyg_objects[i].edge_index.numel() > 1:
            
            targets = example.values[0][1:].astype(float)
            targets = torch.tensor(targets).unsqueeze(-1)
        
        target_names = example.columns.tolist()
        target_names.remove('Drug')
        labeled_ex = pyg_objects[i]
        labeled_ex.y = targets
        labeled_ex.mols = mol

        if '.' in mol:
            print("BAD MOLECULE: {}".format(mol))

        labeled_ex.target_names = target_names

        if labeled_ex.edge_index.size(0) == 0:
            self_ids = torch.range(0, labeled_ex.num_nodes - 1).long().unsqueeze(0)
            self_loops = torch.cat((self_ids, self_ids), dim=0)
            labeled_ex.edge_index = self_loops
        
        else:
            labeled_ex.edge_index = remove_self_loops(labeled_ex.edge_index)[0]
            if contains_isolated_nodes(labeled_ex.edge_index, num_nodes=labeled_ex.num_nodes):
                edge_index, _, mask = remove_isolated_nodes(labeled_ex.edge_index, num_nodes=labeled_ex.num_nodes)
                labeled_ex.x = labeled_ex.x[mask]
                labeled_ex.edge_index = edge_index
                labeled_ex.num_nodes = labeled_ex.x.shape[0]
            labeled_ex.edge_index = add_self_loops(labeled_ex.edge_index)[0]

        if labeled_ex.num_nodes > 1:
            labeled_data.append(labeled_ex.to(cuda))
        else:
            print("Skipping: {}".format(mol))

    return labeled_data

# Save dataset
def save_pyg_dataset(args, labeled_data, set_type) -> None:

    dataset_to_use = 'DEV' if args.dev_mode else 'FULL'

    # torch.save(Dataset(labeled_data), 'data/processed/{}_{}_data_pyg.pt'.format(args.prop_source, set_type))
    with open('data/processed/{}_{}_{}_data_pyg_{}.pkl'.format(args.data_source, args.prop_source, set_type, dataset_to_use), "wb") as file:
        pickle.dump(labeled_data, file)
