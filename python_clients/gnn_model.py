import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import List, Dict, Any, Tuple
import warnings


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self._weight_shapes_cache = None

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def _get_weight_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self._weight_shapes_cache is None:
            self._weight_shapes_cache = {}
            for name, param in self.named_parameters():
                self._weight_shapes_cache[name] = tuple(param.shape)
        return self._weight_shapes_cache
    
    def get_weights(self) -> List[Dict[str, Any]]:
        weights = []
        for name, param in self.named_parameters():
            data = param.data.cpu().numpy()
            
            if not torch.isfinite(param.data).all():
                warnings.warn(
                    f"Warning: Parameter '{name}' contains NaN or Inf values. "
                    f"These will be replaced with zeros."
                )
                data = torch.where(
                    torch.isfinite(param.data),
                    param.data,
                    torch.zeros_like(param.data)
                ).cpu().numpy()
            
            weights.append({
                "name": name,
                "shape": list(data.shape),
                "data": data.flatten().tolist()
            })
        return weights
    
    def set_weights(self, weights_list: List[Dict[str, Any]]) -> bool:
        state_dict = self.state_dict()
        expected_shapes = self._get_weight_shapes()
        
        weight_dict = {}
        for w in weights_list:
            name = w["name"]
            weight_dict[name] = w
        
        applied_count = 0
        errors = []
        
        for name, expected_shape in expected_shapes.items():
            if name not in weight_dict:
                errors.append(f"Missing weight tensor: {name}")
                continue
            
            w = weight_dict[name]
            
            shape = tuple(w.get("shape", []))
            data_list = w.get("data", [])
            
            expected_size = 1
            for s in expected_shape:
                expected_size *= s
            
            if len(data_list) != expected_size:
                errors.append(
                    f"Weight '{name}' size mismatch: expected {expected_size} elements, "
                    f"got {len(data_list)}"
                )
                continue
            
            try:
                data_np = torch.tensor(data_list, dtype=state_dict[name].dtype)
                
                if not torch.isfinite(data_np).all():
                    warnings.warn(
                        f"Warning: Received invalid values (NaN/Inf) for parameter '{name}'. "
                        f"Keeping original weights."
                    )
                    continue
                
                if expected_shape:
                    data_np = data_np.view(expected_shape)
                
                if data_np.shape != state_dict[name].shape:
                    errors.append(
                        f"Weight '{name}' shape mismatch: expected {state_dict[name].shape}, "
                        f"got {data_np.shape}"
                    )
                    continue
                
                state_dict[name] = data_np
                applied_count += 1
                
            except Exception as e:
                errors.append(f"Error loading weight '{name}': {str(e)}")
                continue
        
        if errors:
            for err in errors:
                warnings.warn(err)
        
        try:
            self.load_state_dict(state_dict, strict=False)
            self._weight_shapes_cache = None
            return applied_count == len(expected_shapes)
        except Exception as e:
            warnings.warn(f"Failed to load state dict: {str(e)}")
            return False
    
    def validate_weights(self, weights_list: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        expected_shapes = self._get_weight_shapes()
        errors = []
        
        weight_dict = {}
        for w in weights_list:
            name = w.get("name", "")
            if name:
                weight_dict[name] = w
        
        for name, expected_shape in expected_shapes.items():
            if name not in weight_dict:
                errors.append(f"Missing weight tensor: {name}")
                continue
            
            w = weight_dict[name]
            
            if "data" not in w:
                errors.append(f"Weight '{name}' missing 'data' field")
                continue
            
            data_list = w["data"]
            expected_size = 1
            for s in expected_shape:
                expected_size *= s
            
            if len(data_list) != expected_size:
                errors.append(
                    f"Weight '{name}' size mismatch: expected {expected_size} elements, "
                    f"got {len(data_list)}"
                )
                continue
        
        for name in weight_dict.keys():
            if name not in expected_shapes:
                errors.append(f"Unexpected weight tensor: {name}")
        
        return len(errors) == 0, errors
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
