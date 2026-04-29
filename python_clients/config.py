class Config:
    AGGREGATION_SERVER_URL = "http://localhost:8080"
    
    DATASET_NAME = "Cora"
    NUM_CLIENTS = 3
    HIDDEN_DIM = 64
    NUM_EPOCHS_PER_ROUND = 5
    LEARNING_RATE = 0.01
    DROPOUT = 0.5
    
    CLIENT_API_BASE_PORT = 5000
