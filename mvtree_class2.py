from os import XATTR_REPLACE
from joblib import Parallel, delayed
import pandas as pd
from sktime.transformations.base import _PanelToTabularTransformer
import numpy as np

class _MVTreeExtractor(_PanelToTabularTransformer):
  """Base adapter class for transformations"""
  
  def __init__(
    self,
    time_features = [],
    lags = list(),
    window_functions = dict(),
    exclude_features = dict(),
    n_jobs = -1,
    _is_fitted = False,
    ts_values = None,
    ts_uid = None,
    ts_date = None
    #,
    #categorical_features=dict()
  ):
    self.time_features = time_features
    self.lags = lags
    self.window_functions = window_functions
    #self.categorical_features = categorical_features.copy()
    self.exclude_features = exclude_features
    self.n_jobs = n_jobs
    self._is_fitted = _is_fitted
    self.ts_values = ts_values
    self.ts_uid = ts_uid
    self.ts_date = ts_date
    
    super(_MVTreeExtractor, self).__init__()
        
  ## Get extraction parameters ##    
  def fit(self, X, y=None):
    if len(self.window_functions) > 0:
      _window_functions = list()
      for func_name,rw_config in self.window_functions.items():
          func_call,window_shifts,window_sizes = rw_config
          for window_shift in window_shifts:
              for window_size in window_sizes:
                  _window_functions.append((func_name, func_call, window_shift, window_size))
      self._window_functions = _window_functions
    else:
      self._window_functions = list()

    self._is_fitted = True

    return self

  
class MVTreeFeatureExtractor(_MVTreeExtractor):
    """Transformer for extracting time series features
    """
    
    def transform(self, X, y=None):
      """Transform X.
      Parameters
      ----------
      X : pd.DataFrame
          nested pandas DataFrame of shape [n_samples, n_columns]
      y : pd.Series, optional (default=None)
      Returns
      -------
      Xt : pandas DataFrame
        Transformed pandas DataFrame
      """
      # input checks

      self.check_is_fitted()
      all_features_list = list()
      all_features_list.append(X.reset_index(drop=True))
      
      # generating the time features
      if len(self.time_features) > 0:
        input_params = {"date_range":pd.DatetimeIndex(XATTR_REPLACE[self.ts_date]),
                        "time_features":self.time_features,
                        "ignore_const_cols":False}
        calendar_features = compute_calendar_features(**input_params)
        all_features_list.append(calendar_features)

        # generating the lag & rolling window features
        #if (len(lags) > 0) or 
        if (len(self.window_functions) > 0):
          #  lag_kwargs = [{"lag":lag} for lag in lags]  
            rw_kwargs =  [{"func_name":window_func[0],
                           "func_call":window_func[1], 
                           "window_shift":window_func[2], 
                           "window_size":window_func[3]}
                           for window_func in self._window_functions]
            input_kwargs = rw_kwargs# lag_kwargs + rw_kwargs

            grouped =  X.loc[:, self.ts_uid+[self.ts_values]].groupby(self.ts_uid)[self.ts_values]
            with Parallel(n_jobs=self.n_jobs) as parallel:
                delayed_func = delayed(compute_lagged_train_feature)
                lagged_features = parallel(delayed_func(grouped, **kwargs) for kwargs in input_kwargs)
                lagged_features = pd.DataFrame({feature.name:feature.values for feature in lagged_features})
                all_features_list.append(lagged_features)
        
        

        all_features = pd.concat(all_features_list, axis=1)
        # merging all features
        X.drop(self.exclude_features, axis=1,inplace=True)

        all_features.set_index(X.index, inplace=True)

        return(all_features.drop([self.ts_date, self.ts_values], axis = 1))
      
      
      
def compute_calendar_features(date_range, time_features, ignore_const_cols=True):
  time_features_mapping = {"year_week":"weekofyear",
                     "year_day":"dayofyear",
                     "month_day":"day",
                     "week_day":"dayofweek"}
  """
  Parameters
  ----------
  date_range: pandas.DatetimeIndex or pandas.TimedeltaIndex
      Ranges of date times.
  time_features: List
      Time attributes to include as features.
  ignore_const_cols: bool
      Specify whether to ignore constant columns.
  """  
  calendar_data = pd.DataFrame()

  for feature in time_features:
      if feature in time_features_mapping.keys():
          _feature = time_features_mapping[feature]
      else:
          _feature = feature

      if hasattr(date_range, _feature):
          feature_series = getattr(date_range, _feature)
          if feature_series.nunique() == 1 and ignore_const_cols: 
              continue
          calendar_data[feature] = feature_series

  # other time features
  if "month_progress" in time_features:
      calendar_data["month_progress"] = date_range.day/date_range.days_in_month
  if "millisecond" in time_features:
      calendar_data["millisecond"] = date_range.microsecond//1000

  # cyclical time features
  #https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca
  #Open issue: Information is provided in two columns, splitting tree will not combine information
  if "second_cos" in time_features:
      calendar_data["second_cos"] = np.cos(date_range.second*(2.*np.pi/60))
  if "second_sin" in time_features:        
      calendar_data["second_sin"] = np.sin(date_range.second*(2.*np.pi/60))
  if "minute_cos" in time_features:
      calendar_data["minute_cos"] = np.cos(date_range.minute*(2.*np.pi/60))
  if "minute_sin" in time_features:
      calendar_data["minute_sin"] = np.sin(date_range.minute*(2.*np.pi/60))
  if "hour_cos" in time_features:
      calendar_data["hour_cos"] = np.cos(date_range.hour*(2.*np.pi/24))
  if "hour_sin" in time_features:
      calendar_data["hour_sin"] = np.sin(date_range.hour*(2.*np.pi/24))
  if "week_day_cos" in time_features:
      calendar_data["week_day_cos"] = np.cos(date_range.dayofweek*(2.*np.pi/7))
  if "week_day_sin" in time_features:
      calendar_data["week_day_sin"] = np.sin(date_range.dayofweek*(2.*np.pi/7))
  if "year_day_cos" in time_features:
      calendar_data["year_day_cos"] = np.cos((date_range.dayofyear-1)*(2.*np.pi/366))
  if "year_day_sin" in time_features:
      calendar_data["year_day_sin"] = np.sin((date_range.dayofyear-1)*(2.*np.pi/366))
  ## pandas.DatetimeIndex.weekofyear is depracated (nochmal checken, ob neue Methode richtig funktioniert und so sinnvoll ist) ## 
#       if "year_week_cos" in time_features:
#           calendar_data["year_week_cos"] = np.cos((date_range.weekofyear-1)*(2.*np.pi/52))
#       if "year_week_sin" in time_features:
#           calendar_data["year_week_sin"] = np.sin((date_range.weekofyear-1)*(2.*np.pi/52))
  if "year_week_cos" in time_features:
      calendar_data["year_week_cos"] = np.cos((date_range.isocalendar().week.reset_index(drop = True)-1)*(2.*np.pi/52))
  if "year_week_sin" in time_features:
      calendar_data["year_week_sin"] = np.sin((date_range.isocalendar().week.reset_index(drop = True)-1)*(2.*np.pi/52))
  if "month_cos" in time_features:
      calendar_data["month_cos"] = np.cos((date_range.month-1)*(2.*np.pi/12))
  if "month_sin" in time_features:
      calendar_data["month_sin"] = np.sin((date_range.month-1)*(2.*np.pi/12))

  # week_day shifted to 1-7
  if "week_day" in calendar_data.columns:
      calendar_data["week_day"] += 1

  return calendar_data

def compute_lagged_train_feature(grouped, lag=None, func_name=None, func_call=None, window_shift=None, window_size=None):
  """
  grouped: pandas.core.groupby.generic.SeriesGroupBy
      Groupby object containing the response variable "y"
      grouped by ts_uid_columns.
  lag: int
      Integer lag value.
  func_name: string
      Name of the rolling window function.
  func_call: function or None
      Callable if a custom function, None otherwise.
  window_shift: int
      Integer window shift value.
  window_size: int
      Integer window size value.
  """
  is_lag_feature = lag is not None
  is_rw_feature = (func_name is not None) and (window_shift is not None) and (window_size is not None)
  #feature_values = grouped.apply(lambda x: getattr(x.shift(window_shift).rolling(window_size), "kurt")())

  if is_lag_feature and not is_rw_feature:
      feature_values = grouped.shift(lag)
      feature_values.name = f"lag{lag}"
  elif is_rw_feature and not is_lag_feature:
      if func_call is None:
          # native pandas method
          feature_values = grouped.apply(lambda x: getattr(x.shift(window_shift).rolling(window_size), func_name)())
      else:
          # custom function
          feature_values = grouped.apply(lambda x: x.shift(window_shift).rolling(window_size).apply(func_call, raw=True))
      feature_values.name = f"{func_name}{window_size}_shift{window_shift}"
  else:
      raise ValueError("Invalid input parameters.")

    


  return feature_values