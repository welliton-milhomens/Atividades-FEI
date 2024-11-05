# cliente.py
import flwr as fl
from tensorflow import keras

# Definir o modelo de rede neural
def get_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Carregar e preparar os dados do MNIST
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

# Definir a estratégia do cliente
class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):  # Adicionado argumento 'config'
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Função principal para iniciar o cliente
def main():
    # Carregar dados e modelo
    model = get_model()
    (x_train, y_train), (x_test, y_test) = load_data()

    # Iniciar cliente Flower com a nova função
    client = MnistClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_client(
        server_address="localhost:4588",  # Certifique-se de que o cliente está na mesma porta do servidor
        client=client.to_client()
    )

# Executar a função principal
if __name__ == "__main__":
    main()