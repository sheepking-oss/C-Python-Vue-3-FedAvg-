import threading
import time
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from gnn_model import GCN
from data_loader import FederatedDataLoader
from trainer import FederatedTrainer
from config import Config


class ClientAPIServer:
    def __init__(self, client_id: str, port: int, data_loader: FederatedDataLoader, client_idx: int):
        self.client_id = client_id
        self.port = port
        self.data_loader = data_loader
        self.client_idx = client_idx
        
        self.data, self.sample_count = data_loader.get_client_data(client_idx)
        self.model = None
        self.trainer = None
        
        self.app = Flask(__name__)
        CORS(self.app)
        
        self.is_training = False
        self.current_round = 0
        self.max_rounds = 0
        self.loss_updates = []
        self.lock = threading.Lock()
        
        self._setup_routes()
        self._init_model()
    
    def _init_model(self):
        in_channels = self.data.num_features
        out_channels = self.data_loader.dataset.num_classes
        
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=Config.HIDDEN_DIM,
            out_channels=out_channels,
            dropout=Config.DROPOUT
        )
        
        self.trainer = FederatedTrainer(
            client_id=self.client_id,
            model=self.model,
            data=self.data,
            sample_count=self.sample_count,
            server_url=Config.AGGREGATION_SERVER_URL,
            lr=Config.LEARNING_RATE,
            epochs_per_round=Config.NUM_EPOCHS_PER_ROUND
        )
    
    def _setup_routes(self):
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            with self.lock:
                return jsonify({
                    'client_id': self.client_id,
                    'port': self.port,
                    'is_training': self.is_training,
                    'current_round': self.current_round,
                    'max_rounds': self.max_rounds,
                    'sample_count': self.sample_count,
                    'lr': self.trainer.lr if self.trainer else Config.LEARNING_RATE
                })
        
        @self.app.route('/api/train', methods=['POST'])
        def start_training():
            data = request.json
            
            rounds = data.get('rounds', 10)
            lr = data.get('learning_rate', Config.LEARNING_RATE)
            
            with self.lock:
                if self.is_training:
                    return jsonify({
                        'status': 'error',
                        'message': 'Training already in progress'
                    }), 400
                
                self.is_training = True
                self.max_rounds = rounds
                self.current_round = 0
                self.loss_updates = []
            
            if self.trainer:
                self.trainer.set_learning_rate(lr)
            
            thread = threading.Thread(target=self._run_training, args=(rounds,), daemon=True)
            thread.start()
            
            return jsonify({
                'status': 'success',
                'message': f'Training started for {rounds} rounds',
                'learning_rate': lr,
                'rounds': rounds
            })
        
        @self.app.route('/api/stop', methods=['POST'])
        def stop_training():
            with self.lock:
                if not self.is_training:
                    return jsonify({
                        'status': 'warning',
                        'message': 'No training in progress'
                    })
                
                self.is_training = False
            
            return jsonify({
                'status': 'success',
                'message': 'Training stopped'
            })
        
        @self.app.route('/api/losses', methods=['GET'])
        def get_losses(self=self):
            with self.lock:
                return jsonify({
                    'client_id': self.client_id,
                    'loss_updates': self.loss_updates.copy()
                })
        
        @self.app.route('/api/evaluate', methods=['GET'])
        def evaluate():
            if self.trainer:
                metrics = self.trainer.evaluate()
                return jsonify(metrics)
            return jsonify({'error': 'Trainer not initialized'}), 500
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset():
            with self.lock:
                self.is_training = False
                self.current_round = 0
                self.loss_updates = []
                self._init_model()
            
            return jsonify({
                'status': 'success',
                'message': 'Client reset successfully'
            })
        
        @self.app.route('/api/weights', methods=['GET'])
        def get_weights():
            if self.model:
                weights = self.model.get_weights()
                return jsonify({
                    'client_id': self.client_id,
                    'weights': weights
                })
            return jsonify({'error': 'Model not initialized'}), 500
    
    def _on_loss_update(self, loss_data):
        with self.lock:
            self.loss_updates.append(loss_data)
    
    def _run_training(self, rounds: int):
        print(f"Client {self.client_id} - Starting federated training for {rounds} rounds")
        
        for r in range(1, rounds + 1):
            with self.lock:
                if not self.is_training:
                    print(f"Client {self.client_id} - Training stopped at round {r}")
                    break
                
                self.current_round = r
            
            if r > 1:
                print(f"Client {self.client_id} - Downloading global weights for round {r}")
                self.trainer.download_global_weights()
            
            result = self.trainer.train_round(r, on_loss_update=self._on_loss_update)
            
            avg_loss = result.get('avg_loss')
            valid_epochs = result.get('valid_epochs', 0)
            
            if avg_loss is not None:
                print(f"Client {self.client_id} - Round {r} completed. Valid epochs: {valid_epochs}, Avg Loss: {avg_loss:.4f}")
            else:
                print(f"Client {self.client_id} - Round {r} completed with no valid epochs.")
            
            if valid_epochs > 0:
                print(f"Client {self.client_id} - Uploading weights to aggregation server")
                upload_success = self.trainer.upload_weights(r)
                if not upload_success:
                    print(f"Client {self.client_id} - Warning: Failed to upload weights for round {r}")
            else:
                print(f"Client {self.client_id} - Skipping weight upload (no valid training)")
            
            time.sleep(0.5)
        
        with self.lock:
            self.is_training = False
        
        print(f"Client {self.client_id} - Training completed")
    
    def run(self):
        print(f"========================================")
        print(f"  Client API Server: {self.client_id}")
        print(f"========================================")
        print(f"Starting on http://localhost:{self.port}")
        print(f"Available endpoints:")
        print(f"  GET  /api/status   - Get client status")
        print(f"  POST /api/train    - Start training")
        print(f"  POST /api/stop     - Stop training")
        print(f"  GET  /api/losses   - Get loss history")
        print(f"  GET  /api/evaluate - Evaluate model")
        print(f"  POST /api/reset    - Reset client")
        print(f"========================================")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


def start_client_server(client_id: str, port: int, client_idx: int):
    data_loader = FederatedDataLoader(
        dataset_name=Config.DATASET_NAME,
        num_clients=Config.NUM_CLIENTS,
        non_iid=True
    )
    data_loader.load_dataset()
    data_loader.partition_data()
    
    server = ClientAPIServer(client_id, port, data_loader, client_idx)
    server.run()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python client_api_server.py <client_id> <port> <client_idx>")
        print("Example: python client_api_server.py client_1 5001 0")
        sys.exit(1)
    
    client_id = sys.argv[1]
    port = int(sys.argv[2])
    client_idx = int(sys.argv[3])
    
    start_client_server(client_id, port, client_idx)
