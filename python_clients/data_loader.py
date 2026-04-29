import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import numpy as np
from typing import List, Tuple, Optional
import warnings


class FederatedDataLoader:
    def __init__(
        self, 
        dataset_name: str = "Cora", 
        num_clients: int = 3, 
        non_iid: bool = True,
        min_samples_per_client: int = 5,
        allow_partial_classes: bool = True
    ):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.non_iid = non_iid
        self.min_samples_per_client = min_samples_per_client
        self.allow_partial_classes = allow_partial_classes
        self.dataset = None
        self.client_datasets = None
        
    def load_dataset(self) -> Planetoid:
        print(f"Loading {self.dataset_name} dataset...")
        self.dataset = Planetoid(root='./data', name=self.dataset_name)
        data = self.dataset[0]
        
        num_train = data.train_mask.sum().item()
        num_test = data.test_mask.sum().item()
        num_val = data.val_mask.sum().item()
        
        print(f"Dataset loaded: {self.dataset.num_features} features, {self.dataset.num_classes} classes")
        print(f"  Total nodes: {data.num_nodes}")
        print(f"  Train samples: {num_train}")
        print(f"  Test samples: {num_test}")
        print(f"  Validation samples: {num_val}")
        print(f"  Number of edges: {data.edge_index.shape[1]}")
        
        return self.dataset
    
    def partition_data(self) -> List[Tuple[Data, int]]:
        if self.dataset is None:
            self.load_dataset()
        
        data = self.dataset[0]
        num_nodes = data.num_nodes
        labels = data.y.numpy()
        
        print(f"\nPartitioning data for {self.num_clients} clients...")
        print(f"Partition mode: {'Non-IID (class-based)' if self.non_iid else 'IID (random)'}")
        
        if self.non_iid:
            client_datasets = self._partition_non_iid_safe(data, labels, num_nodes)
        else:
            client_datasets = self._partition_iid_safe(data, labels, num_nodes)
        
        for i, (_, count) in enumerate(client_datasets):
            if count < self.min_samples_per_client:
                warnings.warn(
                    f"Client {i + 1} has only {count} training samples, "
                    f"which is below the recommended minimum of {self.min_samples_per_client}"
                )
        
        self.client_datasets = client_datasets
        
        total_train = sum(count for _, count in client_datasets)
        print(f"\nPartition complete. Total training samples across clients: {total_train}")
        
        return client_datasets
    
    def _partition_iid_safe(self, data: Data, labels: np.ndarray, num_nodes: int) -> List[Tuple[Data, int]]:
        all_train_indices = np.where(data.train_mask.numpy())[0]
        
        if len(all_train_indices) == 0:
            raise ValueError("No training samples available in the dataset")
        
        np.random.shuffle(all_train_indices)
        
        nodes_per_client = max(1, len(all_train_indices) // self.num_clients)
        
        client_datasets = []
        for i in range(self.num_clients):
            start = i * nodes_per_client
            if i < self.num_clients - 1:
                end = start + nodes_per_client
            else:
                end = len(all_train_indices)
            
            client_train_indices = all_train_indices[start:end]
            
            if len(client_train_indices) == 0:
                warnings.warn(f"Client {i + 1} has no training samples in IID partition")
                client_train_indices = all_train_indices[:1]
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[client_train_indices] = True
            
            client_data = self._create_client_data(
                data, num_nodes, train_mask,
                preserve_edges=True,
                include_test_val=True
            )
            
            client_sample_count = train_mask.sum().item()
            client_datasets.append((client_data, client_sample_count))
            
            print(f"Client {i + 1}: {client_sample_count} training samples (IID)")
        
        return client_datasets
    
    def _partition_non_iid_safe(self, data: Data, labels: np.ndarray, num_nodes: int) -> List[Tuple[Data, int]]:
        num_classes = self.dataset.num_classes
        all_train_indices = np.where(data.train_mask.numpy())[0]
        all_train_labels = labels[all_train_indices]
        
        if len(all_train_indices) == 0:
            raise ValueError("No training samples available in the dataset")
        
        class_indices = {}
        for c in range(num_classes):
            class_mask = all_train_labels == c
            class_indices[c] = all_train_indices[class_mask]
            count = len(class_indices[c])
            if count > 0:
                print(f"  Class {c}: {count} training samples")
        
        client_datasets = []
        used_global_indices = set()
        
        for i in range(self.num_clients):
            classes_per_client = max(1, num_classes // self.num_clients)
            
            available_classes = [
                c for c in range(num_classes) 
                if len(class_indices[c]) > 0
            ]
            
            if not available_classes:
                available_classes = [c for c in range(num_classes)]
            
            selected_classes = np.random.choice(
                available_classes,
                size=min(classes_per_client, len(available_classes)),
                replace=False
            )
            
            client_train_indices = []
            for c in selected_classes:
                available_for_class = [
                    idx for idx in class_indices[c] 
                    if idx not in used_global_indices
                ]
                
                if len(available_for_class) > 0:
                    if self.allow_partial_classes and i < self.num_clients - 1:
                        n_take = max(1, len(available_for_class) // 2)
                        selected = np.random.choice(
                            available_for_class,
                            size=min(n_take, len(available_for_class)),
                            replace=False
                        )
                    else:
                        selected = available_for_class
                    
                    client_train_indices.extend(selected)
                    for idx in selected:
                        used_global_indices.add(idx)
            
            if len(client_train_indices) == 0:
                warnings.warn(f"Client {i + 1}: No samples available for selected classes, using fallback allocation")
                unused = [idx for idx in all_train_indices if idx not in used_global_indices]
                if unused:
                    client_train_indices = unused[:max(1, len(unused) // 2)]
                    for idx in client_train_indices:
                        used_global_indices.add(idx)
                else:
                    client_train_indices = list(all_train_indices)[:max(1, len(all_train_indices) // self.num_clients)]
            
            client_train_indices = np.array(client_train_indices)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[client_train_indices] = True
            
            client_data = self._create_client_data(
                data, num_nodes, train_mask,
                preserve_edges=True,
                include_test_val=True
            )
            
            client_sample_count = train_mask.sum().item()
            client_datasets.append((client_data, client_sample_count))
            
            actual_classes = np.unique(labels[client_train_indices])
            print(f"Client {i + 1}: {client_sample_count} training samples, Classes: {actual_classes} (Non-IID)")
        
        return client_datasets
    
    def _create_client_data(
        self,
        original_data: Data,
        num_nodes: int,
        train_mask: torch.Tensor,
        preserve_edges: bool = True,
        include_test_val: bool = True
    ) -> Data:
        if include_test_val:
            test_mask = original_data.test_mask & ~train_mask
            val_mask = original_data.val_mask & ~train_mask
        else:
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        edge_index = original_data.edge_index
        if not preserve_edges:
            train_node_indices = torch.where(train_mask)[0]
            edge_mask = torch.isin(edge_index[0], train_node_indices) & \
                       torch.isin(edge_index[1], train_node_indices)
            edge_index = edge_index[:, edge_mask]
        
        client_data = Data(
            x=original_data.x,
            edge_index=edge_index,
            y=original_data.y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        
        return client_data
    
    def get_client_data(self, client_idx: int) -> Tuple[Data, int]:
        if self.client_datasets is None:
            self.partition_data()
        
        if 0 <= client_idx < self.num_clients:
            data, count = self.client_datasets[client_idx]
            
            if count == 0:
                warnings.warn(
                    f"Client {client_idx} has 0 training samples. "
                    f"This may cause issues during training."
                )
            
            return data, count
        else:
            raise ValueError(f"Invalid client index: {client_idx}. Must be between 0 and {self.num_clients - 1}")
    
    def get_data_statistics(self) -> dict:
        if self.dataset is None:
            self.load_dataset()
        
        data = self.dataset[0]
        stats = {
            'dataset_name': self.dataset_name,
            'num_nodes': data.num_nodes,
            'num_features': self.dataset.num_features,
            'num_classes': self.dataset.num_classes,
            'num_edges': data.edge_index.shape[1],
            'train_samples': data.train_mask.sum().item(),
            'test_samples': data.test_mask.sum().item(),
            'val_samples': data.val_mask.sum().item(),
            'num_clients': self.num_clients,
            'partition_mode': 'non_iid' if self.non_iid else 'iid'
        }
        
        if self.client_datasets:
            client_stats = []
            for i, (data, count) in enumerate(self.client_datasets):
                client_stats.append({
                    'client_id': i,
                    'train_samples': count,
                    'test_samples': data.test_mask.sum().item(),
                    'val_samples': data.val_mask.sum().item()
                })
            stats['clients'] = client_stats
        
        return stats
