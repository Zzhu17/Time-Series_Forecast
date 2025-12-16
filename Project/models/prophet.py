from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error

def build_prophet(df, yearly=True, weekly=False, daily=False,
                   changepoint_prior_scale=0.5,
                   seasonality_prior_scale=10.0,
                   seasonality_mode='additive',
                   n_changepoints=25,
                   auto_tune=False,
                   val_df=None):

    if auto_tune and val_df is not None:
        param_grid = {
            'changepoint_prior_scale': [0.05, 0.1, 0.3, 0.5],
            'seasonality_prior_scale': [5.0, 10.0, 20.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'n_changepoints': [15, 25, 40]
        }

        best_params = None
        best_mape = float('inf')

        for params in ParameterGrid(param_grid):
            model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=weekly,
                daily_seasonality=daily,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                n_changepoints=params['n_changepoints']
            )
            model.fit(df)

            future_val = model.make_future_dataframe(periods=len(val_df))
            forecast_val = model.predict(future_val)
            y_pred = forecast_val['yhat'][-len(val_df):].values
            y_true = val_df['y'].values

            mape = mean_absolute_percentage_error(y_true, y_pred)
            if mape < best_mape:
                best_mape = mape
                best_params = params

        model = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            changepoint_prior_scale=best_params['changepoint_prior_scale'],
            seasonality_prior_scale=best_params['seasonality_prior_scale'],
            seasonality_mode=best_params['seasonality_mode'],
            n_changepoints=best_params['n_changepoints']
        )
        model.fit(df)
        return model

    else:
        model = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            seasonality_mode=seasonality_mode,
            n_changepoints=n_changepoints
        )
        model.fit(df)
        return model

def build_prophet_model(config):
    return Prophet(
        yearly_seasonality=config.get("yearly_seasonality", True),
        weekly_seasonality=config.get("weekly_seasonality", False),
        daily_seasonality=config.get("daily_seasonality", False),
        seasonality_mode=config.get("seasonality_mode", "additive"),
        changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05)
    )