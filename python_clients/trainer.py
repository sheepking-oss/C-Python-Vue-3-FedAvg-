import torch
import torch.nn.functional as F
import requests
import json
from typing import Dict, List, Optional, Callable
from gnn_model import GCN
from torch_geometric.data import Data


class FederatedTrainer:
    def __init__(
        self,
        client_id: str,
        model: GCN,
        data: Data,
        sample_count: int,
        server_url: str,
        lr: float = 0.01,
        epochs_per_round: int = 5
    ):
        self.client_id = client_id
        self.model = model
        self.data = data
        self.sample_count = sample_count
        self.server_url = server_url
        self.lr = lr
        self.epochs_per_round = epochs_per_round
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        
        self.optimizer = None
        self.loss_history = []
        self.current_round = 1
        
    def _init_optimizer(self):
        if self.optimizer is None or self.optimizer.param_groups[0]['lr'] != self.lr:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def set_learning_rate(self, lr: float):
        self.lr = lr
        self._init_optimizer()
    
    def train_epoch(self) -> float:
        self.model.train()
        self._init_optimizer()
        
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_round(self, round_num: int, on_loss_update: Optional[Callable] = None) -> Dict:
        self.current_round = round_num
        round_losses = []
        
        print(f"Client {self.client_id} - Starting training round {round_num}")
        
        for epoch in range(self.epochs_per_round):
            loss = self.train_epoch()
            round_losses.append(loss)
            
            if on_loss_update:
                on_loss_update({
                    'client_id': self.client_id,
                    'round': round_num,
                    'epoch': epoch + 1,
                    'loss': loss
                })
            
            if (epoch + 1) % 5 == 0:
                print(f"Client {self.client_id} - Round {round_num}, Epoch {epoch + 1}: Loss = {loss:.4f}")
        
        avg_loss = sum(round_losses) / len(round_losses)
        self.loss_history.append({
            'round': round_num,
            'losses': round_losses,
            'avg_loss': avg_loss
        })
        
        return {
            'client_id': self.client_id,
            'round': round_num,
            'avg_loss': avg_loss,
            'losses': round_losses
        }
    
    def evaluate(self) -> Dict:
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            
            train_pred = out[self.data.train_mask].argmax(dim=1)
            train_acc = (train_pred == self.data.y[self.data.train_mask]).sum().item() / self.data.train_mask.sum().item()
            
            test_pred = out[self.data.test_mask].argmax(dim=1)
            test_acc = (test_pred == self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            
            val_pred = out[self.data.val_mask].argmax(dim=1)
            val_acc = (val_pred == self.data.y[self.data.val_mask]).sum().item() / self.data.val_mask.sum().item()
        
        return {
            'client_id': self.client_id,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'val_acc': val_acc
        }
    
    def upload_weights(self, round_num: int) -> bool:
        weights = self.model.get_weights()
        
        payload = {
            'client_id': self.client_id,
            'sample_count': self.sample_count,
            'round': round_num,
            'weights': weights
        }
        
        try:
            response = requests.post(
                f"{self.server_url}/api/submit",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Client {self.client_id} - Successfully uploaded weights to server")
                return True
            else:
                print(f"Client {self.client_id} - Failed to upload weights: {response.text}")
                return False
        except Exception as e:
            print(f"Client {self.client_id} - Error uploading weights: {e}")
            return False
    
    def download_global_weights(self) -> bool:
        try:
            response = requests.get(
                f"{self.server_url}/api/weights",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'weights' in data and data['weights']:
                    self.model.set_weights(data['weights'])
                    print(f"Client {self.client_id} - Successfully downloaded global weights from server")
                    return True
                else:
                    print(f"Client {self.client_id} - No global weights available yet")
                    return False
            else:
                print(f"Client {self.client_id} - Failed to download weights: {response.text}")
                return False
        except Exception as e:
            print(f"Client {self.client_id} - Error downloading weights: {e}")
            return False
    
    def get_loss_history(self) -> List[Dict]:
        return self.loss_history
