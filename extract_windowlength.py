model_kwargs = {
  "time_features":['month',"year_month_sin","year_month_cos"],
  "window_functions":{
    "mean":   (None, [1], [2,8,24]),
    "std":   (None, [1], [6]),
    #"median":   (None, [1], [4]),
    #"czeros": (compute_czeros, [1,4], [1,4,8]), 
    #"std":    (None, [1,4], [2,4,8]),
    #"std":   (None, [1, 4], [2, 12])
  },
  "exclude_features": ["ts_id"],
  "ts_values": "y",
  "ts_uid": ["ts_id"], ## Must be a list
  "ts_date": "Period"
}

def find_maxlag(model_kwargs):
  window_functions = model_kwargs["window_functions"]
  if len(window_functions) > 0:
    _window_functions = list()
    for func_name,rw_config in window_functions.items():
        func_call,window_shifts,window_sizes = rw_config
        for window_shift in window_shifts:
            for window_size in window_sizes:
                _window_functions.append((func_name, func_call, window_shift, window_size))
    _window_functions = _window_functions
  else:
    _window_functions = list()
  if (len(_window_functions) > 0):
    #  lag_kwargs = [{"lag":lag} for lag in lags]  
      rw_kwargs =  [{"func_name":window_func[0],
                      "func_call":window_func[1], 
                      "window_shift":window_func[2], 
                      "window_size":window_func[3]}
                      for window_func in _window_functions]
  collect_length = [(i["window_shift"]-1+i["window_size"]) for i in rw_kwargs]
  collect_max = max(collect_length)
  #y[y.index <(collect_max-1)] = None
  #y=y[(collect_max-1):]
  return(collect_max)

print(find_maxlag(model_kwargs))