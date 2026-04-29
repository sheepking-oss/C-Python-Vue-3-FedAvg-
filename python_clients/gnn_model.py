import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_weights(self):
        weights = []
        for name, param in self.named_parameters():
            weights.append({
                "name": name,
                "shape": list(param.shape),
                "data": param.data.cpu().numpy().flatten().tolist()
            })
        return weights
    
    def set_weights(self, weights_list):
        state_dict = self.state_dict()
        for w in weights_list:
            name = w["name"]
            if name in state_dict:
                tensor = torch.tensor(w["data"], dtype=state_dict[name].dtype)
                if w["shape"]:
                    tensor = tensor.view(w["shape"])
                state_dict[name] = tensor
        self.load_state_dict(state_dict)
