# server.py
import flwr as fl

# Iniciar o servidor Flower
if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:4590",  # Alterar para uma porta diferente
        config=fl.server.ServerConfig(num_rounds=3),
    )