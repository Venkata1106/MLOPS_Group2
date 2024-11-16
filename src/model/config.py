from typing import Dict, Any

MODEL_CONFIG = {
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'training': {
        'test_size': 0.2,
        'random_state': 42
    },
    'stocks': [
        'AAPL', 'ADBE', 'AMZN', 'CSCO', 'GOOGL',
        'META', 'MSFT', 'NVDA', 'PEP', 'TSLA'
    ],
    'data_path': None,  # Set by DAG
    'output_path': None  # Set by DAG
} 