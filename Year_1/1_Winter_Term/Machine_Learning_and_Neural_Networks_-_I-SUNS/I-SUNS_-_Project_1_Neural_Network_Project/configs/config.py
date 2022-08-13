CONFIG = {
    "data": {
        "training_data": "./data/wine_train.csv",
        "testing_data": "./data/wine_test.csv"
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
                },
                "under_train": {
                    "train": {
                        "epochs": 50,
                        "learning_rate": 0.0001,
                        "patience": 0,
                        "batch_size": 3318,
                        "metrics": ["accuracy"]
                    },
                    "model": {
                        "activation_function": "relu",
                        "hidden_layers": 2,
                        "neurons_per_layer": 2
                    }
                },
                "over_train": {
                    "train": {
                        "epochs": 2000,
                        "learning_rate": 0.0001,
                        "patience": 0,
                        "batch_size": 3318,
                        "metrics": ["accuracy"]
                    },
                    "model": {
                        "activation_function": "relu",
                        "hidden_layers": 10,
                        "neurons_per_layer": 256
                    }
                },
                "fast_train": {
                    "train": {
                        "epochs": 2000,
                        "learning_rate": 0.1,
                        "patience": 0,
                        "batch_size": 1024,
                        "metrics": ["accuracy"]
                    },
                    "model": {
                        "activation_function": "relu",
                        "hidden_layers": 3,
                        "neurons_per_layer": 64
                    }
                },
                "slow_train": {
                    "train": {
                        "epochs": 10000,
                        "learning_rate": 0.00000000000000000000000000001,
                        "patience": 0,
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
        }
    }
}
