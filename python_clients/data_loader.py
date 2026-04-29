import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple


class FederatedDataLoader:
    def __init__(self, dataset_name: str = "Cora", num_clients: int = 3, non_iid: bool = True):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.non_iid = non_iid
        self.dataset = None
        self.client_datasets = None
        
    def load_dataset(self) -> Planetoid:
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = Planetoid(root='./data', name=self.dataset_name)
        print(f"Dataset loaded: {len(self.dataset)} graphs, {self.dataset.num_features} features, {self.dataset.num_classes} classes")
        return self.dataset
    
    def partition_data(self) -> List[Tuple[Data, int]]:
        if self.dataset is None:
            self.load_dataset()
        
        data = self.dataset[0]
        num_nodes = data.num_nodes
        labels = data.y.numpy()
        
        if self.non_iid:
            client_datasets = self._partition_non_iid(data, labels, num_nodes)
        else:
            client_datasets = self._partition_iid(data, labels, num_nodes)
        
        self.client_datasets = client_datasets
        return client_datasets
    
    def _partition_iid(self, data: Data, labels: np.ndarray, num_nodes: int) -> List[Tuple[Data, int]]:
        indices = np.random.permutation(num_nodes)
        nodes_per_client = num_nodes // self.num_clients
        
        client_datasets = []
        for i in range(self.num_clients):
            start = i * nodes_per_client
            end = start + nodes_per_client if i < self.num_clients - 1 else num_nodes
            
            client_indices = indices[start:end]
            
            mask = torch.zeros(num_nodes, dtype=torch.bool)
            mask[client_indices] = True
            
            train_mask = mask & data.train_mask
            test_mask = mask & data.test_mask
            val_mask = mask & data.val_mask
            
            client_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                y=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                val_mask=val_mask
            )
            
            client_sample_count = train_mask.sum().item()
            client_datasets.append((client_data, client_sample_count))
            
            print(f"Client {i + 1}: {client_sample_count} training samples (IID)")
        
        return client_datasets
    
    def _partition_non_iid(self, data: Data, labels: np.ndarray, num_nodes: int) -> List[Tuple[Data, int]]:
        num_classes = self.dataset.num_classes
        client_datasets = []
        
        for i in range(self.num_clients):
            selected_classes = np.random.choice(
                num_classes, 
                size=max(1, num_classes // 2), 
                replace=False
            )
            
            class_mask = np.isin(labels, selected_classes)
            all_indices = np.where(class_mask)[0]
            
            train_indices = np.intersect1d(all_indices, np.where(data.train_mask.numpy())[0])
            test_indices = np.intersect1d(all_indices, np.where(data.test_mask.numpy())[0])
            val_indices = np.intersect1d(all_indices, np.where(data.val_mask.numpy())[0])
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[train_indices] = True
            test_mask[test_indices] = True
            val_mask[val_indices] = True
            
            client_data = Data(
                x=data.x,
                edge_index=data.edge_index,
                y=data.y,
                train_mask=train_mask,
                test_mask=test_mask,
                val_mask=val_mask
            )
            
            client_sample_count = train_mask.sum().item()
            client_datasets.append((client_data, client_sample_count))
            
            print(f"Client {i + 1}: {client_sample_count} training samples, Classes: {selected_classes} (Non-IID)")
        
        return client_datasets
    
    def get_client_data(self, client_idx: int) -> Tuple[Data, int]:
        if self.client_datasets is None:
            self.partition_data()
        
        if 0 <= client_idx < self.num_clients:
            return self.client_datasets[client_idx]
        else:
            raise ValueError(f"Invalid client index: {client_idx}")
