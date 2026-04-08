from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

from evaluation.backtest import expanding_window_splits


_PARAM_GRID_TEMPLATES = {
    'small': {
        'changepoint_prior_scale': [0.05, 0.3],
        'seasonality_prior_scale': [10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'n_changepoints': [15, 25],
    },
    'medium': {
        'changepoint_prior_scale': [0.05, 0.1, 0.3],
        'seasonality_prior_scale': [5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'n_changepoints': [15, 25],
    },
    'full': {
        'changepoint_prior_scale': [0.05, 0.1, 0.3, 0.5],
        'seasonality_prior_scale': [5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'n_changepoints': [15, 25, 40],
    },
}


def _resolve_param_grid(template='medium'):
    return _PARAM_GRID_TEMPLATES.get(template, _PARAM_GRID_TEMPLATES['medium'])


def _evaluate_holdout(df, val_df, yearly, weekly, daily, param_grid):
    best_params = None
    best_mape = float('inf')
    best_mae = float('inf')

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
        mae = mean_absolute_error(y_true, y_pred)
        if mape < best_mape:
            best_mape = mape
            best_mae = mae
            best_params = params

    return best_params, {'mode': 'holdout', 'mape': best_mape, 'mae': best_mae, 'fallback_to_holdout': False}


def _evaluate_rolling(df, yearly, weekly, daily, param_grid, min_folds=3):
    n_samples = len(df)
    val_size = max(1, n_samples // 10)
    folds = expanding_window_splits(
        n_samples=n_samples,
        n_splits=min_folds,
        val_size=val_size,
        min_train_size=max(20, val_size * 2),
    )
    if len(folds) < min_folds:
        return None, {'mode': 'holdout', 'fallback_to_holdout': True, 'reason': 'insufficient_samples_for_rolling'}

    best_params = None
    best_mape = float('inf')
    best_mae = float('inf')

    for params in ParameterGrid(param_grid):
        fold_mapes = []
        fold_maes = []
        for train_slice, val_slice in folds:
            train_df = df.iloc[train_slice]
            val_df = df.iloc[val_slice]
            model = Prophet(
                yearly_seasonality=yearly,
                weekly_seasonality=weekly,
                daily_seasonality=daily,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                n_changepoints=params['n_changepoints']
            )
            model.fit(train_df)
            future_val = model.make_future_dataframe(periods=len(val_df))
            forecast_val = model.predict(future_val)
            y_pred = forecast_val['yhat'][-len(val_df):].values
            y_true = val_df['y'].values
            fold_mapes.append(mean_absolute_percentage_error(y_true, y_pred))
            fold_maes.append(mean_absolute_error(y_true, y_pred))

        avg_mape = float(sum(fold_mapes) / len(fold_mapes))
        avg_mae = float(sum(fold_maes) / len(fold_maes))
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_mae = avg_mae
            best_params = params

    return best_params, {
        'mode': 'rolling',
        'mape': best_mape,
        'mae': best_mae,
        'n_folds': len(folds),
        'fallback_to_holdout': False,
    }

def build_prophet(df, yearly=True, weekly=False, daily=False,
                   changepoint_prior_scale=0.5,
                   seasonality_prior_scale=10.0,
                   seasonality_mode='additive',
                   n_changepoints=25,
                   auto_tune=False,
                   val_df=None,
                   cv_mode='rolling',
                   param_grid_template='medium',
                   min_rolling_folds=3):

    if auto_tune and val_df is not None:
        param_grid = _resolve_param_grid(param_grid_template)

        if cv_mode == 'rolling':
            best_params, cv_scores = _evaluate_rolling(
                df=df,
                yearly=yearly,
                weekly=weekly,
                daily=daily,
                param_grid=param_grid,
                min_folds=min_rolling_folds,
            )
            if best_params is None:
                best_params, holdout_scores = _evaluate_holdout(
                    df=df,
                    val_df=val_df,
                    yearly=yearly,
                    weekly=weekly,
                    daily=daily,
                    param_grid=param_grid,
                )
                cv_scores.update(holdout_scores)
                cv_scores['fallback_to_holdout'] = True
        else:
            best_params, cv_scores = _evaluate_holdout(
                df=df,
                val_df=val_df,
                yearly=yearly,
                weekly=weekly,
                daily=daily,
                param_grid=param_grid,
            )

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
        return model, cv_scores, best_params

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
        return model, None, None
