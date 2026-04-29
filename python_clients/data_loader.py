import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, add_self_loops, degree, to_dense_adj
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
import warnings


class SubgraphExtractor:
    @staticmethod
    def extract_subgraph(
        original_data: Data,
        node_indices: torch.Tensor,
        include_self_loops: bool = True,
        relabel_nodes: bool = True
    ) -> Tuple[Data, Dict[int, int]]:
        if node_indices.numel() == 0:
            empty_data = Data(
                x=torch.empty((0, original_data.num_features), dtype=original_data.x.dtype),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.empty(0, dtype=original_data.y.dtype),
                train_mask=torch.empty(0, dtype=torch.bool),
                test_mask=torch.empty(0, dtype=torch.bool),
                val_mask=torch.empty(0, dtype=torch.bool)
            )
            return empty_data, {}
        
        node_indices = node_indices.to(torch.long)
        
        edge_index, edge_attr = subgraph(
            node_indices,
            original_data.edge_index,
            relabel_nodes=relabel_nodes,
            num_nodes=original_data.num_nodes
        )
        
        if include_self_loops and edge_index.size(1) > 0:
            num_subgraph_nodes = node_indices.size(0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_subgraph_nodes)
        
        x = original_data.x[node_indices]
        y = original_data.y[node_indices]
        
        idx_map = {int(original_idx): int(new_idx) for new_idx, original_idx in enumerate(node_indices)}
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=node_indices.size(0)
        ), idx_map
    
    @staticmethod
    def extract_subgraph_with_masks(
        original_data: Data,
        train_node_indices: torch.Tensor,
        test_node_indices: Optional[torch.Tensor] = None,
        val_node_indices: Optional[torch.Tensor] = None,
        include_isolated_nodes: bool = True
    ) -> Tuple[Data, Dict]:
        all_nodes = [train_node_indices]
        if test_node_indices is not None and test_node_indices.numel() > 0:
            all_nodes.append(test_node_indices)
        if val_node_indices is not None and val_node_indices.numel() > 0:
            all_nodes.append(val_node_indices)
        
        if not all_nodes:
            raise ValueError("No nodes provided for subgraph extraction")
        
        combined_nodes = torch.cat(all_nodes).unique()
        
        subgraph_data, idx_map = SubgraphExtractor.extract_subgraph(
            original_data, combined_nodes, relabel_nodes=True
        )
        
        num_subgraph_nodes = subgraph_data.num_nodes
        
        train_mask = torch.zeros(num_subgraph_nodes, dtype=torch.bool)
        for idx in train_node_indices:
            mapped_idx = idx_map.get(int(idx))
            if mapped_idx is not None:
                train_mask[mapped_idx] = True
        
        test_mask = torch.zeros(num_subgraph_nodes, dtype=torch.bool)
        if test_node_indices is not None:
            for idx in test_node_indices:
                mapped_idx = idx_map.get(int(idx))
                if mapped_idx is not None:
                    test_mask[mapped_idx] = True
        
        val_mask = torch.zeros(num_subgraph_nodes, dtype=torch.bool)
        if val_node_indices is not None:
            for idx in val_node_indices:
                mapped_idx = idx_map.get(int(idx))
                if mapped_idx is not None:
                    val_mask[mapped_idx] = True
        
        subgraph_data.train_mask = train_mask
        subgraph_data.test_mask = test_mask
        subgraph_data.val_mask = val_mask
        
        stats = {
            'total_nodes': num_subgraph_nodes,
            'train_nodes': int(train_mask.sum()),
            'test_nodes': int(test_mask.sum()),
            'val_nodes': int(val_mask.sum()),
            'edges': subgraph_data.edge_index.size(1),
            'isolated_nodes': int(
                (degree(subgraph_data.edge_index[0], num_nodes=num_subgraph_nodes) == 0).sum()
            )
        }
        
        return subgraph_data, stats, idx_map


class GraphDataValidator:
    @staticmethod
    def validate_for_gnn(data: Data, min_train_nodes: int = 1) -> Tuple[bool, List[str]]:
        errors = []
        warnings_list = []
        
        if data.x is None or data.x.numel() == 0:
            errors.append("Node features (x) are empty")
        
        if data.y is None or data.y.numel() == 0:
            errors.append("Labels (y) are empty")
        
        if data.edge_index is None:
            errors.append("Edge index is None")
        elif data.edge_index.size(1) == 0:
            if data.num_nodes > 0:
                warnings_list.append(
                    f"Graph has {data.num_nodes} nodes but 0 edges. "
                    f"GNN may not work properly without message passing."
                )
        
        train_count = int(data.train_mask.sum()) if data.train_mask is not None else 0
        if train_count < min_train_nodes:
            errors.append(
                f"Insufficient training nodes: {train_count} < {min_train_nodes}"
            )
        
        if data.x is not None and data.y is not None:
            if data.x.size(0) != data.y.size(0):
                errors.append(
                    f"Feature count ({data.x.size(0)}) != Label count ({data.y.size(0)})"
                )
        
        if not errors and data.edge_index is not None and data.edge_index.size(1) > 0:
            num_nodes = data.num_nodes
            max_edge_idx = int(data.edge_index.max()) if data.edge_index.numel() > 0 else -1
            if max_edge_idx >= num_nodes:
                errors.append(
                    f"Edge index contains node {max_edge_idx} but graph only has {num_nodes} nodes"
                )
        
        return len(errors) == 0, errors + warnings_list
    
    @staticmethod
    def prepare_sparse_graph(data: Data, add_self_loops: bool = True) -> Data:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            if add_self_loops and data.num_nodes > 0:
                node_indices = torch.arange(data.num_nodes, dtype=torch.long)
                self_loops = torch.stack([node_indices, node_indices])
                data = data.clone()
                data.edge_index = self_loops
                warnings.warn(
                    f"Graph has no edges. Added self-loops for {data.num_nodes} nodes."
                )
        
        return data


class FederatedDataLoader:
    def __init__(
        self, 
        dataset_name: str = "Cora", 
        num_clients: int = 3, 
        non_iid: bool = True,
        min_samples_per_client: int = 5,
        allow_partial_classes: bool = True,
        extract_real_subgraph: bool = True,
        add_self_loops_for_isolated: bool = True
    ):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.non_iid = non_iid
        self.min_samples_per_client = min_samples_per_client
        self.allow_partial_classes = allow_partial_classes
        self.extract_real_subgraph = extract_real_subgraph
        self.add_self_loops_for_isolated = add_self_loops_for_isolated
        
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
        print(f"  Total edges: {data.edge_index.shape[1]}")
        print(f"  Train samples: {num_train}")
        print(f"  Test samples: {num_test}")
        print(f"  Validation samples: {num_val}")
        
        valid, issues = GraphDataValidator.validate_for_gnn(data, min_train_nodes=1)
        if not valid:
            warnings.warn(f"Original graph has issues: {issues}")
        
        return self.dataset
    
    def partition_data(self) -> List[Tuple[Data, int]]:
        if self.dataset is None:
            self.load_dataset()
        
        data = self.dataset[0]
        num_nodes = data.num_nodes
        labels = data.y.numpy()
        
        print(f"\nPartitioning data for {self.num_clients} clients...")
        print(f"Partition mode: {'Non-IID (class-based)' if self.non_iid else 'IID (random)'}")
        print(f"Subgraph extraction: {'Enabled (real subgraphs)' if self.extract_real_subgraph else 'Disabled (full graph with masks)'}")
        
        if self.non_iid:
            client_datasets = self._partition_non_iid_safe(data, labels, num_nodes)
        else:
            client_datasets = self._partition_iid_safe(data, labels, num_nodes)
        
        for i, (client_data, count) in enumerate(client_datasets):
            valid, issues = GraphDataValidator.validate_for_gnn(
                client_data, min_train_nodes=min(1, self.min_samples_per_client)
            )
            
            if self.add_self_loops_for_isolated:
                client_datasets[i] = (
                    GraphDataValidator.prepare_sparse_graph(client_data, add_self_loops=True),
                    count
                )
            
            if not valid:
                warnings.warn(f"Client {i + 1} data issues: {issues}")
            
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
            
            if self.extract_real_subgraph:
                client_data, stats, _ = self._extract_client_subgraph(
                    data, client_train_indices, i
                )
                client_sample_count = stats['train_nodes']
            else:
                client_data, client_sample_count = self._create_masked_data(
                    data, num_nodes, client_train_indices
                )
            
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
                if len([idx for idx in class_indices[c] if idx not in used_global_indices]) > 0
            ]
            
            if not available_classes:
                available_classes = [c for c in range(num_classes) if len(class_indices[c]) > 0]
            
            if not available_classes:
                available_classes = list(range(num_classes))
            
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
            
            if self.extract_real_subgraph:
                client_data, stats, _ = self._extract_client_subgraph(
                    data, client_train_indices, i
                )
                client_sample_count = stats['train_nodes']
            else:
                client_data, client_sample_count = self._create_masked_data(
                    data, num_nodes, client_train_indices
                )
            
            client_datasets.append((client_data, client_sample_count))
            
            actual_classes = np.unique(labels[client_train_indices])
            print(f"Client {i + 1}: {client_sample_count} training samples, Classes: {actual_classes} (Non-IID)")
        
        return client_datasets
    
    def _extract_client_subgraph(
        self,
        original_data: Data,
        train_indices: np.ndarray,
        client_idx: int
    ) -> Tuple[Data, Dict, Dict]:
        train_tensor = torch.tensor(train_indices, dtype=torch.long)
        
        all_train_indices = np.where(original_data.train_mask.numpy())[0]
        all_test_indices = np.where(original_data.test_mask.numpy())[0]
        all_val_indices = np.where(original_data.val_mask.numpy())[0]
        
        nearby_test = []
        nearby_val = []
        
        edge_index = original_data.edge_index.numpy()
        train_set = set(train_indices)
        
        for src, dst in zip(edge_index[0], edge_index[1]):
            if src in train_set and dst not in train_set:
                if dst in all_test_indices:
                    nearby_test.append(dst)
                elif dst in all_val_indices:
                    nearby_val.append(dst)
            elif dst in train_set and src not in train_set:
                if src in all_test_indices:
                    nearby_test.append(src)
                elif src in all_val_indices:
                    nearby_val.append(src)
        
        nearby_test = np.array(list(set(nearby_test)))
        nearby_val = np.array(list(set(nearby_val)))
        
        if len(nearby_test) == 0:
            nearby_test = all_test_indices[:min(10, len(all_test_indices))]
        if len(nearby_val) == 0:
            nearby_val = all_val_indices[:min(5, len(all_val_indices))]
        
        test_tensor = torch.tensor(nearby_test, dtype=torch.long) if len(nearby_test) > 0 else None
        val_tensor = torch.tensor(nearby_val, dtype=torch.long) if len(nearby_val) > 0 else None
        
        try:
            subgraph_data, stats, idx_map = SubgraphExtractor.extract_subgraph_with_masks(
                original_data,
                train_tensor,
                test_tensor,
                val_tensor,
                include_isolated_nodes=True
            )
            
            print(f"  Client {client_idx + 1} subgraph: {stats['total_nodes']} nodes, {stats['edges']} edges")
            if stats['isolated_nodes'] > 0:
                print(f"    Note: {stats['isolated_nodes']} isolated nodes (degree 0)")
            
            return subgraph_data, stats, idx_map
            
        except Exception as e:
            warnings.warn(f"Subgraph extraction failed for client {client_idx + 1}: {e}. Using masked data instead.")
            return self._create_masked_data_with_stats(
                original_data, 
                original_data.num_nodes, 
                train_indices
            )
    
    def _create_masked_data(
        self,
        original_data: Data,
        num_nodes: int,
        train_indices: np.ndarray
    ) -> Tuple[Data, int]:
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        
        test_mask = original_data.test_mask & ~train_mask
        val_mask = original_data.val_mask & ~train_mask
        
        client_data = Data(
            x=original_data.x,
            edge_index=original_data.edge_index,
            y=original_data.y,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=val_mask
        )
        
        client_sample_count = train_mask.sum().item()
        return client_data, client_sample_count
    
    def _create_masked_data_with_stats(
        self,
        original_data: Data,
        num_nodes: int,
        train_indices: np.ndarray
    ) -> Tuple[Data, Dict, Dict]:
        client_data, sample_count = self._create_masked_data(
            original_data, num_nodes, train_indices
        )
        
        stats = {
            'total_nodes': num_nodes,
            'train_nodes': sample_count,
            'test_nodes': int(original_data.test_mask.sum()),
            'val_nodes': int(original_data.val_mask.sum()),
            'edges': original_data.edge_index.size(1),
            'isolated_nodes': 0
        }
        
        idx_map = {i: i for i in range(num_nodes)}
        
        return client_data, stats, idx_map
    
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
            
            valid, issues = GraphDataValidator.validate_for_gnn(data, min_train_nodes=1)
            if not valid:
                warnings.warn(f"Client {client_idx} data validation issues: {issues}")
            
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
            'partition_mode': 'non_iid' if self.non_iid else 'iid',
            'subgraph_extraction': self.extract_real_subgraph
        }
        
        if self.client_datasets:
            client_stats = []
            for i, (data, count) in enumerate(self.client_datasets):
                valid, issues = GraphDataValidator.validate_for_gnn(data, min_train_nodes=1)
                client_stats.append({
                    'client_id': i,
                    'train_samples': count,
                    'test_samples': int(data.test_mask.sum()) if data.test_mask is not None else 0,
                    'val_samples': int(data.val_mask.sum()) if data.val_mask is not None else 0,
                    'total_nodes': data.num_nodes,
                    'num_edges': data.edge_index.size(1) if data.edge_index is not None else 0,
                    'valid': valid,
                    'issues': issues
                })
            stats['clients'] = client_stats
        
        return stats
