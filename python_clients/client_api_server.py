import threading
import time
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from gnn_model import GCN
from data_loader import FederatedDataLoader, GraphDataValidator
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
        
        self._validate_data()
        self._setup_routes()
        self._init_model()
    
    def _validate_data(self):
        print(f"[{self.client_id}] Validating client data...")
        
        valid, issues = GraphDataValidator.validate_for_gnn(
            self.data, 
            min_train_nodes=max(1, Config.MIN_SAMPLES_PER_CLIENT)
        )
        
        if not valid:
            print(f"[{self.client_id}] WARNING: Data validation issues: {issues}")
        else:
            print(f"[{self.client_id}] Data validation passed")
        
        stats = {
            'nodes': self.data.num_nodes,
            'edges': self.data.edge_index.size(1) if self.data.edge_index is not None else 0,
            'train_samples': self.sample_count,
            'features': self.data.num_features
        }
        print(f"[{self.client_id}] Data stats: {stats}")
        
        if self.data.edge_index is not None and self.data.edge_index.numel() > 0:
            max_edge_idx = int(self.data.edge_index.max())
            if max_edge_idx >= self.data.num_nodes:
                print(f"[{self.client_id}] CRITICAL: Edge index out of bounds!")
                print(f"  Max edge idx: {max_edge_idx}, Num nodes: {self.data.num_nodes}")
    
    def _init_model(self):
        num_features = self.data.num_features
        num_classes = self.data_loader.dataset.num_classes
        
        print(f"[{self.client_id}] Initializing GCN model...")
        print(f"  Input features: {num_features}")
        print(f"  Hidden dim: {Config.HIDDEN_DIM}")
        print(f"  Output classes: {num_classes}")
        
        self.model = GCN(
            in_channels=num_features,
            hidden_channels=Config.HIDDEN_DIM,
            out_channels=num_classes,
            dropout=Config.DROPOUT
        )
        
        num_params = self.model.get_num_parameters()
        print(f"[{self.client_id}] Model initialized with {num_params} parameters")
        
        self.trainer = FederatedTrainer(
            client_id=self.client_id,
            model=self.model,
            data=self.data,
            sample_count=self.sample_count,
            server_url=Config.AGGREGATION_SERVER_URL,
            lr=Config.LEARNING_RATE,
            epochs_per_round=Config.NUM_EPOCHS_PER_ROUND,
            min_train_samples=Config.MIN_SAMPLES_PER_CLIENT
        )
    
    def _setup_routes(self):
        @self.app.route('/api/status', methods=['GET'])
        def get_status():
            with self.lock:
                trainer_status = {}
                if self.trainer:
                    trainer_status = self.trainer.get_status()
                
                return jsonify({
                    'client_id': self.client_id,
                    'port': self.port,
                    'is_training': self.is_training,
                    'current_round': self.current_round,
                    'max_rounds': self.max_rounds,
                    'sample_count': self.sample_count,
                    'lr': self.trainer.lr if self.trainer else Config.LEARNING_RATE,
                    'can_train': self.trainer.can_train() if self.trainer else False,
                    'data_stats': {
                        'num_nodes': self.data.num_nodes,
                        'num_edges': self.data.edge_index.size(1) if self.data.edge_index is not None else 0,
                        'num_features': self.data.num_features
                    },
                    'trainer_status': trainer_status
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
                
                if self.trainer and not self.trainer.can_train():
                    warnings.warn(f"[{self.client_id}] Trainer reports cannot train, but attempting anyway")
                
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
                'rounds': rounds,
                'can_train': self.trainer.can_train() if self.trainer else True
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
                    'weights': weights,
                    'num_weights': len(weights)
                })
            return jsonify({'error': 'Model not initialized'}), 500
        
        @self.app.route('/api/data_stats', methods=['GET'])
        def get_data_stats():
            stats = self.data_loader.get_data_statistics()
            return jsonify(stats)
    
    def _on_loss_update(self, loss_data):
        with self.lock:
            self.loss_updates.append(loss_data)
    
    def _run_training(self, rounds: int):
        print(f"[{self.client_id}] ========================================")
        print(f"[{self.client_id}] Starting federated training for {rounds} rounds")
        print(f"[{self.client_id}] ========================================")
        
        for r in range(1, rounds + 1):
            with self.lock:
                if not self.is_training:
                    print(f"[{self.client_id}] Training stopped at round {r}")
                    break
                
                self.current_round = r
            
            print(f"[{self.client_id}] --- Round {r}/{rounds} ---")
            
            if r > 1:
                print(f"[{self.client_id}] Downloading global weights...")
                download_success = self.trainer.download_global_weights()
                if not download_success:
                    print(f"[{self.client_id}] Warning: Failed to download global weights, using local weights")
            
            print(f"[{self.client_id}] Training locally...")
            result = self.trainer.train_round(r, on_loss_update=self._on_loss_update)
            
            avg_loss = result.get('avg_loss')
            valid_epochs = result.get('valid_epochs', 0)
            
            if avg_loss is not None:
                print(f"[{self.client_id}] Round {r} complete. Valid epochs: {valid_epochs}, Avg Loss: {avg_loss:.4f}")
            else:
                print(f"[{self.client_id}] Round {r} complete. No valid epochs.")
            
            if valid_epochs > 0 and self.sample_count >= Config.MIN_SAMPLES_PER_CLIENT:
                print(f"[{self.client_id}] Uploading weights to aggregation server...")
                upload_success = self.trainer.upload_weights(r)
                if upload_success:
                    print(f"[{self.client_id}] Weights uploaded successfully")
                else:
                    print(f"[{self.client_id}] Warning: Failed to upload weights")
            else:
                print(f"[{self.client_id}] Skipping weight upload (insufficient valid training)")
            
            print(f"[{self.client_id}] Waiting before next round...")
            time.sleep(0.5)
        
        with self.lock:
            self.is_training = False
        
        print(f"[{self.client_id}] ========================================")
        print(f"[{self.client_id}] Training completed")
        print(f"[{self.client_id}] ========================================")
    
    def run(self):
        print(f"========================================")
        print(f"  Client API Server: {self.client_id}")
        print(f"========================================")
        print(f"Starting on http://localhost:{self.port}")
        print(f"Configuration:")
        print(f"  - Dataset: {Config.DATASET_NAME}")
        print(f"  - Min samples per client: {Config.MIN_SAMPLES_PER_CLIENT}")
        print(f"  - Non-IID partition: {Config.NON_IID_PARTITION}")
        print(f"  - Extract real subgraph: {Config.EXTRACT_REAL_SUBGRAPH}")
        print(f"  - Add self-loops for isolated: {Config.ADD_SELF_LOOPS_FOR_ISOLATED}")
        print(f"Available endpoints:")
        print(f"  GET  /api/status     - Get client status")
        print(f"  POST /api/train      - Start training")
        print(f"  POST /api/stop       - Stop training")
        print(f"  GET  /api/losses     - Get loss history")
        print(f"  GET  /api/evaluate   - Evaluate model")
        print(f"  POST /api/reset      - Reset client")
        print(f"  GET  /api/weights    - Get current weights")
        print(f"  GET  /api/data_stats - Get data statistics")
        print(f"========================================")
        
        self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)


def start_client_server(client_id: str, port: int, client_idx: int):
    print(f"\n{'='*50}")
    print(f"Initializing {client_id}...")
    print(f"{'='*50}")
    
    data_loader = FederatedDataLoader(
        dataset_name=Config.DATASET_NAME,
        num_clients=Config.NUM_CLIENTS,
        non_iid=Config.NON_IID_PARTITION,
        min_samples_per_client=Config.MIN_SAMPLES_PER_CLIENT,
        allow_partial_classes=Config.ALLOW_PARTIAL_CLASSES,
        extract_real_subgraph=Config.EXTRACT_REAL_SUBGRAPH,
        add_self_loops_for_isolated=Config.ADD_SELF_LOOPS_FOR_ISOLATED
    )
    
    print(f"\n[{client_id}] Loading and partitioning data...")
    data_loader.load_dataset()
    data_loader.partition_data()
    
    stats = data_loader.get_data_statistics()
    print(f"\n[{client_id}] Data statistics:")
    print(f"  Total nodes: {stats['num_nodes']}")
    print(f"  Total edges: {stats['num_edges']}")
    print(f"  Features: {stats['num_features']}")
    print(f"  Classes: {stats['num_classes']}")
    
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
