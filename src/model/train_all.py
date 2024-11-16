from model_development import StockPredictionModel
from config import MODEL_CONFIG
import logging
from scripts.logger_config import setup_logging

def main():
    setup_logging()
    model = StockPredictionModel(MODEL_CONFIG)
    
    for symbol in MODEL_CONFIG['stocks']:
        try:
            logging.info(f"Training model for {symbol}")
            model.train_model(symbol)
            logging.info(f"Successfully trained model for {symbol}")
        except Exception as e:
            logging.error(f"Error training model for {symbol}: {str(e)}")

if __name__ == "__main__":
    main() 