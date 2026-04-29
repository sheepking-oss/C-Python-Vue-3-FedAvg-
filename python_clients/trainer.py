import torch
import torch.nn.functional as F
import requests
import json
from typing import Dict, List, Optional, Callable, Tuple
from gnn_model import GCN
from torch_geometric.data import Data
import warnings


class FederatedTrainer:
    def __init__(
        self,
        client_id: str,
        model: GCN,
        data: Data,
        sample_count: int,
        server_url: str,
        lr: float = 0.01,
        epochs_per_round: int = 5,
        min_train_samples: int = 1
    ):
        self.client_id = client_id
        self.model = model
        self.data = data
        self.sample_count = sample_count
        self.server_url = server_url
        self.lr = lr
        self.epochs_per_round = epochs_per_round
        self.min_train_samples = min_train_samples
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        self.optimizer = None
        self.loss_history = []
        self.current_round = 1
        self._is_initialized = False
        
        self._validate_and_prepare()
        
    def _validate_and_prepare(self):
        print(f"[{self.client_id}] Validating training setup...")
        
        train_count = self.data.train_mask.sum().item()
        print(f"[{self.client_id}] Training samples: {train_count}")
        
        if train_count < self.min_train_samples:
            warnings.warn(
                f"[{self.client_id}] Only {train_count} training samples available. "
                f"Minimum recommended: {self.min_train_samples}"
            )
        
        try:
            test_count = self.data.test_mask.sum().item()
            val_count = self.data.val_mask.sum().item()
            print(f"[{self.client_id}] Test samples: {test_count}, Validation samples: {val_count}")
        except:
            pass
        
        num_nodes = self.data.num_nodes
        num_edges = self.data.edge_index.shape[1] if self.data.edge_index is not None else 0
        print(f"[{self.client_id}] Graph: {num_nodes} nodes, {num_edges} edges")
        
        num_params = self.model.get_num_parameters()
        print(f"[{self.client_id}] Model parameters: {num_params}")
        
        self._is_initialized = True
        print(f"[{self.client_id}] Setup complete. Device: {self.device}")
        
    def _init_optimizer(self):
        if self.optimizer is None or self.optimizer.param_groups[0]['lr'] != self.lr:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            print(f"[{self.client_id}] Optimizer initialized with learning rate: {self.lr}")
    
    def set_learning_rate(self, lr: float):
        if lr <= 0:
            warnings.warn(f"[{self.client_id}] Invalid learning rate: {lr}, using 0.01")
            lr = 0.01
        self.lr = lr
        self._init_optimizer()
    
    def _safe_loss_calculation(
        self, 
        out: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], int]:
        mask_count = mask.sum().item()
        
        if mask_count == 0:
            return None, 0
        
        if mask_count < self.min_train_samples:
            warnings.warn(
                f"[{self.client_id}] Only {mask_count} samples for loss calculation. "
                f"Proceeding but results may be unstable."
            )
        
        try:
            loss = F.nll_loss(out[mask], self.data.y[mask])
            
            if not torch.isfinite(loss):
                warnings.warn(f"[{self.client_id}] Loss is NaN or Inf, returning None")
                return None, mask_count
            
            return loss, mask_count
            
        except Exception as e:
            warnings.warn(f"[{self.client_id}] Error during loss calculation: {str(e)}")
            return None, mask_count
    
    def train_epoch(self) -> Optional[float]:
        train_count = self.data.train_mask.sum().item()
        
        if train_count < self.min_train_samples:
            warnings.warn(
                f"[{self.client_id}] Insufficient training samples: {train_count}. "
                f"Need at least {self.min_train_samples}."
            )
            return None
        
        try:
            self.model.train()
            self._init_optimizer()
            
            self.optimizer.zero_grad()
            
            out = self.model(self.data.x, self.data.edge_index)
            
            loss, used_samples = self._safe_loss_calculation(out, self.data.train_mask)
            
            if loss is None:
                warnings.warn(f"[{self.client_id}] Failed to compute loss, skipping epoch")
                return None
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            loss_value = loss.item()
            
            if not (0 <= loss_value <= 1e10):
                warnings.warn(f"[{self.client_id}] Unusual loss value: {loss_value}")
                return None
            
            return loss_value
            
        except Exception as e:
            warnings.warn(f"[{self.client_id}] Error during training epoch: {str(e)}")
            return None
    
    def train_round(
        self, 
        round_num: int, 
        on_loss_update: Optional[Callable] = None
    ) -> Dict:
        self.current_round = round_num
        round_losses = []
        valid_epochs = 0
        
        print(f"[{self.client_id}] Starting training round {round_num}")
        print(f"[{self.client_id}] Epochs per round: {self.epochs_per_round}")
        
        for epoch in range(self.epochs_per_round):
            loss = self.train_epoch()
            
            if loss is not None:
                round_losses.append(loss)
                valid_epochs += 1
                
                if on_loss_update:
                    on_loss_update({
                        'client_id': self.client_id,
                        'round': round_num,
                        'epoch': epoch + 1,
                        'loss': loss
                    })
                
                if (epoch + 1) % max(1, self.epochs_per_round // 5) == 0:
                    print(f"[{self.client_id}] Round {round_num}, Epoch {epoch + 1}: Loss = {loss:.4f}")
            else:
                warnings.warn(f"[{self.client_id}] Round {round_num}, Epoch {epoch + 1}: Failed to compute loss")
        
        if valid_epochs == 0:
            warnings.warn(f"[{self.client_id}] No valid epochs in round {round_num}")
            avg_loss = None
        else:
            avg_loss = sum(round_losses) / len(round_losses)
            print(f"[{self.client_id}] Round {round_num} complete. Valid epochs: {valid_epochs}, Avg Loss: {avg_loss:.4f}")
        
        self.loss_history.append({
            'round': round_num,
            'losses': round_losses,
            'avg_loss': avg_loss,
            'valid_epochs': valid_epochs
        })
        
        return {
            'client_id': self.client_id,
            'round': round_num,
            'avg_loss': avg_loss,
            'losses': round_losses,
            'valid_epochs': valid_epochs,
            'total_epochs': self.epochs_per_round
        }
    
    def evaluate(self) -> Dict:
        self.model.eval()
        
        results = {
            'client_id': self.client_id,
            'train_acc': None,
            'test_acc': None,
            'val_acc': None,
            'train_samples': 0,
            'test_samples': 0,
            'val_samples': 0
        }
        
        try:
            with torch.no_grad():
                out = self.model(self.data.x, self.data.edge_index)
                
                for mask_name, mask in [
                    ('train', self.data.train_mask),
                    ('test', self.data.test_mask),
                    ('val', self.data.val_mask)
                ]:
                    sample_count = mask.sum().item()
                    results[f'{mask_name}_samples'] = sample_count
                    
                    if sample_count > 0:
                        pred = out[mask].argmax(dim=1)
                        correct = (pred == self.data.y[mask]).sum().item()
                        acc = correct / sample_count
                        results[f'{mask_name}_acc'] = acc
                        print(f"[{self.client_id}] {mask_name.capitalize()} accuracy: {acc:.4f} ({correct}/{sample_count})")
                    else:
                        warnings.warn(f"[{self.client_id}] No {mask_name} samples available for evaluation")
                        
        except Exception as e:
            warnings.warn(f"[{self.client_id}] Error during evaluation: {str(e)}")
        
        return results
    
    def upload_weights(self, round_num: int) -> bool:
        train_count = self.data.train_mask.sum().item()
        
        if train_count < self.min_train_samples:
            warnings.warn(
                f"[{self.client_id}] Not uploading weights: insufficient training samples "
                f"({train_count} < {self.min_train_samples})"
            )
            return False
        
        try:
            weights = self.model.get_weights()
            
            num_weights = len(weights)
            total_elements = sum(len(w['data']) for w in weights)
            print(f"[{self.client_id}] Preparing to upload {num_weights} tensors ({total_elements} elements)")
            
            effective_sample_count = max(train_count, 1)
            
            payload = {
                'client_id': self.client_id,
                'sample_count': effective_sample_count,
                'round': round_num,
                'weights': weights
            }
            
            print(f"[{self.client_id}] Uploading weights to {self.server_url}/api/submit...")
            
            response = requests.post(
                f"{self.server_url}/api/submit",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"[{self.client_id}] Successfully uploaded weights. Server response: {result.get('status', 'success')}")
                return True
            else:
                error_msg = response.text[:500] if response.text else "Unknown error"
                print(f"[{self.client_id}] Failed to upload weights. Status: {response.status_code}")
                print(f"[{self.client_id}] Error: {error_msg}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[{self.client_id}] Timeout while uploading weights")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[{self.client_id}] Connection error while uploading weights")
            return False
        except Exception as e:
            print(f"[{self.client_id}] Error uploading weights: {str(e)}")
            return False
    
    def download_global_weights(self) -> bool:
        try:
            print(f"[{self.client_id}] Downloading global weights from {self.server_url}/api/weights...")
            
            response = requests.get(
                f"{self.server_url}/api/weights",
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'weights' in data and data['weights']:
                    weights = data['weights']
                    
                    is_valid, errors = self.model.validate_weights(weights)
                    
                    if not is_valid:
                        warnings.warn(
                            f"[{self.client_id}] Received invalid weights from server: {errors}"
                        )
                        return False
                    
                    success = self.model.set_weights(weights)
                    
                    if success:
                        print(f"[{self.client_id}] Successfully downloaded and applied global weights (round {data.get('round', 'unknown')})")
                        return True
                    else:
                        warnings.warn(f"[{self.client_id}] Failed to apply some weights")
                        return False
                else:
                    print(f"[{self.client_id}] No global weights available yet")
                    return False
            else:
                print(f"[{self.client_id}] Failed to download weights. Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"[{self.client_id}] Error downloading weights: {str(e)}")
            return False
    
    def get_loss_history(self) -> List[Dict]:
        return self.loss_history
    
    def can_train(self) -> bool:
        train_count = self.data.train_mask.sum().item()
        return train_count >= self.min_train_samples
    
    def get_status(self) -> Dict:
        return {
            'client_id': self.client_id,
            'device': str(self.device),
            'can_train': self.can_train(),
            'train_samples': self.data.train_mask.sum().item(),
            'test_samples': self.data.test_mask.sum().item(),
            'val_samples': self.data.val_mask.sum().item(),
            'current_round': self.current_round,
            'learning_rate': self.lr,
            'epochs_per_round': self.epochs_per_round,
            'num_parameters': self.model.get_num_parameters(),
            'is_initialized': self._is_initialized
        }
