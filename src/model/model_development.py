import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
import logging
from fairlearn.metrics import MetricFrame
import joblib
from src.model.config import MODEL_CONFIG


# Update imports to match your existing scripts
from scripts.bias_detection import BiasAnalyzer  # Changed from BiasDetector
from scripts.bias_mitigation import BiasMitigator

class StockPredictionModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = GradientBoostingRegressor(**config[MODEL_CONFIG['model_params']])
        self.metrics = {}
        self.bias_analyzer = BiasAnalyzer()  # Changed from BiasDetector
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load data for a specific stock from the mitigated dataset"""
        try:
            file_path = os.path.join(self.config['data_path'], f'mitigated_{symbol}.csv')
            data = pd.read_csv(file_path)
            logging.info(f"Loaded data for {symbol}: {len(data)} records")
            return data
        except Exception as e:
            logging.error(f"Error loading data for {symbol}: {str(e)}")
            raise
            
    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare features and target"""
        # Check if 'date' column exists
        if 'Date' in data.columns:
            # Convert 'date' column to datetime
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            # Drop rows with invalid dates
            data = data.dropna(subset=['Date'])
            # Extract date features
            data['Day'] = data['Date'].dt.day
            data['Month'] = data['Date'].dt.month
            data['Year'] = data['Date'].dt.year
            data['Day_of_Week'] = data['Date'].dt.dayofweek
            # Drop the original date column if not needed
            data = data.drop(columns=['Date'])
        else:
            logging.error("Date column is missing from the data.")
            raise ValueError("Date column is missing from the data.")
        
        # Define features and target
        features = ['Day','Month','Year','Day_of_Week','Open','High','Low','Volume','Returns','MA5','MA20','Volatility']
        target = 'Close'
        
        # Check if the expected columns are in the DataFrame
        missing_columns = [col for col in features if col not in data.columns]
        if missing_columns:
            logging.error(f"Missing columns in data: {missing_columns}")
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        X = data[features]
        y = data[target]
        
        return train_test_split(
            X, y,
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state']
        )
        
    def train_model(self, data: pd.DataFrame) -> dict:
        """Train the model and return metrics"""
        try:
            # Prepare data and get train/test split
            X_train, X_test, y_train, y_test = self.prepare_data(data)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            logging.info(f"Model metrics: {metrics}")
            logging.info(f"Model {self.model}")
            
            # Return both metrics and the test data for future use
            return {
                'status': 'success',
                'metrics': metrics,
                'test_data': {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            }
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def evaluate_model_bias(self, symbol: str) -> dict:
        """Evaluate model bias across different data slices"""
        try:
            # Load the actual model instead of metrics
            model_file = os.path.join(self.config['output_path'], f'{symbol}_model.pkl')
            self.model = joblib.load(model_file)  # Load the actual model object
            
            # Load and prepare data
            data = self.load_data(symbol)
            X_train, X_test, y_train, y_test = self.prepare_data(data)
            
            # Validate model
            validation_metrics = self.validate_model(X_test, y_test)
            
            # Add bias metrics if needed
            bias_report = self.bias_analyzer.generate_report()
            
            return {
                'status': 'success',
                'validation_metrics': validation_metrics,
                'bias_metrics': bias_report
            }
            
        except Exception as e:
            logging.error(f"Error evaluating bias for {symbol}: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
            
    def _save_metrics(self, symbol: str, metrics: dict):
        """Save model metrics"""
        output_dir = 'data/model_metrics'
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'{symbol}_metrics.json')
        with open(file_path, 'w') as f:
            json.dump(metrics, f)
            
    def _save_model(self, symbol: str):
        """Save trained model"""
        import joblib
        
        output_dir = 'data/models'
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'{symbol}_model.joblib')
        joblib.dump(self.model, file_path)
        
    def _save_bias_metrics(self, symbol: str, metrics: dict):
        """Save bias metrics"""
        output_dir = 'data/model_metrics'
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'{symbol}_bias_metrics.json')
        with open(file_path, 'w') as f:
            json.dump(metrics, f)

    def train_and_select_model(self, symbol: str) -> dict:
        """Train the model and select the best configuration"""
        try:
            data = self.load_data(symbol)
            X_train, X_test, y_train, y_test = self.prepare_data(data)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'r2': r2
            }
            
            logging.info(f"Model metrics for {symbol}: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error training model for {symbol}: {str(e)}")
            raise

    def validate_model(self, X_test, y_test) -> dict:
        """Validate the model on a hold-out dataset"""
 
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        validation_metrics = {
            'mse': mse,
            'r2': r2
        }
        
        logging.info(f"Validation metrics: {validation_metrics}")
        return validation_metrics

    def detect_bias(self, X_test, y_test, sensitive_features) -> dict:
        """Detect bias using slicing techniques"""
        y_pred = self.model.predict(X_test)
        
        # Example using Fairlearn
        metric_frame = MetricFrame(
            metrics=mean_squared_error,
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        bias_report = metric_frame.by_group
        logging.info(f"Bias report: {bias_report}")
        return bias_report

    def check_for_bias(self, symbol: str):
        """Check for bias in the model predictions"""
        data = self.load_data(symbol)
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        # Assume 'sensitive_feature' is a column in your dataset
        sensitive_features = X_test['sensitive_feature']
        
        bias_report = self.detect_bias(X_test, y_test, sensitive_features)
        return bias_report

    def save_model(self, symbol: str):
        """Save the trained model to a file"""
        output_dir = os.path.join(self.config['output_path'], 'models')
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, f'{symbol}_model.joblib')
        joblib.dump(self.model, file_path)
        logging.info(f"Model saved for {symbol} at {file_path}")
