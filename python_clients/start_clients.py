import subprocess
import sys
import os
import time
from multiprocessing import Process
from config import Config


def run_client(client_id, port, client_idx):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    client_script = os.path.join(script_dir, 'client_api_server.py')
    
    subprocess.run([
        sys.executable, 
        client_script, 
        client_id, 
        str(port), 
        str(client_idx)
    ])


def main():
    num_clients = Config.NUM_CLIENTS
    base_port = Config.CLIENT_API_BASE_PORT
    
    print("========================================")
    print("  Starting Federated Learning Clients")
    print("========================================")
    print(f"Number of clients: {num_clients}")
    print(f"Aggregation server: {Config.AGGREGATION_SERVER_URL}")
    print("========================================")
    
    processes = []
    
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        port = base_port + i + 1
        client_idx = i
        
        print(f"Starting {client_id} on port {port}...")
        
        p = Process(target=run_client, args=(client_id, port, client_idx))
        p.start()
        processes.append(p)
        
        time.sleep(1)
    
    print("\nAll clients started!")
    print("\nClient endpoints:")
    for i in range(num_clients):
        print(f"  client_{i+1}: http://localhost:{base_port + i + 1}")
    
    print("\nPress Ctrl+C to stop all clients...")
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nStopping all clients...")
        for p in processes:
            p.terminate()
            p.join()
        print("All clients stopped.")


if __name__ == '__main__':
    main()
