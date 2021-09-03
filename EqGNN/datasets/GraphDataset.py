import os
from os.path import join
import warnings
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import dgl


from EqGNN.constants import PROJECT_ROOT
from EqGNN.constants import CACHE_PATH


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, sensitive_attribute):
        super().__init__()
        self.dataset_name = dataset_name
        self.sensitive_attribute = sensitive_attribute

    def prepare_data(self):
        file_path = os.path.join(CACHE_PATH, f'dataset-{self.dataset_name};{self.sensitive_attribute}')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as input:
                self.graph_dgl, self.attribute2mask, self.mask = pickle.load(input)
            return

        graph_nx = get_dataset_graph(self.dataset_name, sensitive_attribute=self.sensitive_attribute)

        node2attribute = nx.get_node_attributes(graph_nx, 'labels')
        enc = OneHotEncoder()
        enc.fit(np.array(list(node2attribute.values())).reshape(-1, 1))
        node2attribute = {node: enc.transform([[node2attribute[node]]]).toarray()[0] for node in node2attribute}
        nx.set_node_attributes(graph_nx, node2attribute, 'labels')
        attribute2NULL = {'labels': enc.transform([['None']]).toarray()[0]}

        node2attribute = nx.get_node_attributes(graph_nx, 'sensitive_attribute')
        nx.set_node_attributes(graph_nx, {node: [node2attribute[node]] for node in node2attribute}, 'sensitive_attribute')

        self.graph_dgl = dgl.from_networkx(graph_nx, node_attrs=['labels', 'sensitive_attribute', 'features'])

        nodes = self.graph_dgl.nodes()
        is_node_null = lambda node: (np.array_equal(self.graph_dgl.nodes[node].data['labels'].numpy()[0], attribute2NULL['labels']))
        self.mask = np.array([False if is_node_null(node) else True for node in nodes])

        self.attribute2mask = {'labels': attribute2NULL['labels'] == 0}

        with open(file_path, 'wb') as output:
            pickle.dump((self.graph_dgl, self.attribute2mask, self.mask), output, pickle.HIGHEST_PROTOCOL)

    def setup(self, stage=None):
        mask_indices = np.where(self.mask)[0]
        np.random.shuffle(mask_indices)

        train_indices = mask_indices[: int(len(mask_indices) * 0.5)]
        train_mask = np.zeros_like(self.mask, dtype=bool)
        train_mask[train_indices] = True
        self.train = GraphDataset(self.graph_dgl, self.attribute2mask, train_mask)

        val_indices = mask_indices[int(len(mask_indices) * 0.5): int(len(mask_indices) * 0.75)]
        val_mask = np.zeros_like(self.mask, dtype=bool)
        val_mask[val_indices] = True
        self.val = GraphDataset(self.graph_dgl, self.attribute2mask, val_mask)

        test_indices = mask_indices[int(len(mask_indices) * 0.75):]
        test_mask = np.zeros_like(self.mask, dtype=bool)
        test_mask[test_indices] = True
        self.test = GraphDataset(self.graph_dgl, self.attribute2mask, test_mask)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

class GraphDataset(Dataset):
    def __init__(self, graph_dgl, attribute2mask, mask):
        self.graph_dgl = graph_dgl
        self.attribute2mask = attribute2mask
        self.mask = mask

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'graph': self.graph_dgl,
                'features': self.graph_dgl.ndata['features'],
                'labels': self.graph_dgl.ndata['labels'],
                'sensitive_attribute': self.graph_dgl.ndata['sensitive_attribute'],
                'mask': self.mask,
                'attribute2mask': self.attribute2mask}


def get_dataset_graph(dataset_name, nrows=None, sensitive_attribute=None):
    '''
    This function is responsible of receiving the dataset name and access the right folder, read and create the graph.
    Args:
        dataset_name: str - the name of the graph dataset
        nrows: int - number of rows to real. Default None - read all rows
        sensitive_attribute: str - relevant only for Pokec. Should be 'region' or 'gender'

    Returns:
        graph_nx: networkx - graph of the dataset with all needed attributes
    '''
    folder_path = join(PROJECT_ROOT, "data", dataset_name)

    if dataset_name == 'pokec':
        if sensitive_attribute is None or (sensitive_attribute!= 'gender' and sensitive_attribute!='region'):
            warnings.warn('pokec must receive a sensitive_attribute: \'gender\' or \'region\'. Using \'gender\'.')
            sensitive_attribute = 'gender'

        graph_path = join(folder_path, "region_job_relationship.txt")
        graph_df = pd.read_table(graph_path, names=['from', 'to'], nrows=nrows)
        graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', create_using=nx.Graph())

        user_df = pd.read_table(join(folder_path, 'region_job.csv'), delimiter=',', nrows=nrows)
        user_df = user_df[user_df['user_id'].isin(graph_nx.nodes())]
        graph_nx = nx.subgraph(graph_nx, user_df['user_id'].unique())
        occupation_map = {0: 0., 1: 1., 2: 0., 3: 2., 4: 1.}# 0-student; 1-services and trade; 2-education; 3-unemployed; 4-construction
        node2occupation = {row['user_id']: str(occupation_map[row['I_am_working_in_field']]) if row['I_am_working_in_field'] != -1 else 'None' for _, row in user_df.iterrows()}
        nx.set_node_attributes(graph_nx, node2occupation, 'labels')

        node2sensitive = {row['user_id']: row[sensitive_attribute] for _, row in user_df.iterrows()}
        nx.set_node_attributes(graph_nx, node2sensitive, 'sensitive_attribute')

        feature_columns = [column for column in user_df.columns if column not in ['user_id', sensitive_attribute, 'I_am_working_in_field']]
        node2features = {row['user_id']: row[feature_columns].tolist() for _, row in user_df.iterrows()}
        nx.set_node_attributes(graph_nx, node2features, 'features')
    elif dataset_name == 'NBA':
        graph_path = join(folder_path, "nba_relationship.txt")
        graph_df = pd.read_table(graph_path, names=['from', 'to'], nrows=nrows)
        graph_df['from'] = graph_df['from'].astype(str)
        graph_df['to'] = graph_df['to'].astype(str)
        graph_nx = nx.from_pandas_edgelist(graph_df, 'from', 'to', create_using=nx.Graph())

        user_df = pd.read_table(join(folder_path, 'nba.csv'), delimiter=',', nrows=nrows)
        user_df['user_id'] = user_df['user_id'].astype(int).astype(str)
        user_df = user_df[user_df['user_id'].isin(graph_nx.nodes())]
        graph_nx = nx.subgraph(graph_nx, user_df['user_id'].unique())
        node2SALARY = {row['user_id']: str(row['SALARY']) if row['SALARY'] != -1 else 'None' for _, row in user_df.iterrows()}
        node2SALARY = {node: node2SALARY[node] if node in node2SALARY else 'None' for node in graph_nx.nodes}
        nx.set_node_attributes(graph_nx, node2SALARY, 'labels')
        node2country = {row['user_id']: row['country'] for _, row in user_df.iterrows()}
        nx.set_node_attributes(graph_nx, node2country, 'sensitive_attribute')

        feature_columns = [column for column in user_df.columns if column not in ['user_id', 'country', 'SALARY']]
        node2features = {row['user_id']: row[feature_columns].tolist() for _, row in user_df.iterrows()}
        nx.set_node_attributes(graph_nx, node2features, 'features')
    else:
        raise Exception('dataset not available')

    return graph_nx
