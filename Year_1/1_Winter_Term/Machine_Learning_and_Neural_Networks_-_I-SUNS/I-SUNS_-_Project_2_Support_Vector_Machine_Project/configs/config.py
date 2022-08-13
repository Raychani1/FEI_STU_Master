CONFIG = {
    "data": {
        "training_data": "./data/spotify_train.csv",
        "testing_data": "./data/spotify_test.csv"
    },
    "models": {
        "NeuralNetwork": {
            "type": {
                "best": {
                    "train": {
                        "epochs": 10000,
                        "learning_rate": 0.001,
                        "patience": 25,
                        "batch_size": 1024,
                        "metrics": ["accuracy"]
                    },
                    "model": {
                        "activation_function": "relu",
                        "hidden_layers": 3,
                        "neurons_per_layer": 64
                    }
                }
            }
        },
    }
}
