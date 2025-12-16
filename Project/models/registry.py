from training.adaptor.informer_adaptor import train_informer_model_7tuple as train_informer_model
from training.adaptor.Prophet_adaptor import train_prophet_model_7tuple as train_prophet_model
from training.adaptor.arima_adaptor  import train_arima_model_7tuple as train_arima_model
from training.adaptor.LSTM_adaptor   import train_lstm_model_7tuple as train_lstm_model
from training.train_random_forest import train_random_forest_model
from models.informer.informer import build_informer_model
from models.arima import build_auto_arima
from models.prophet import build_prophet
from models.random_forest import build_random_forest
from models.lstm import lstm_model

MODEL_REGISTRY = {
    "arima": build_auto_arima,
    "prophet": build_prophet,
    "randomforest": build_random_forest,
    "informer": build_informer_model,
    "lstm": lstm_model,
}

TRAINER_REGISTRY = {
    "informer": train_informer_model,   
    "prophet":  train_prophet_model,    
    "arima":    train_arima_model,      
    "randomforest": train_random_forest_model,
    "lstm":     train_lstm_model,
}

__all__ = ("MODEL_REGISTRY", "TRAINER_REGISTRY")
