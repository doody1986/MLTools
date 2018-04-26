import numpy as np
from scipy.stats.stats import pearsonr

def Calc(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  temp = pearsonr(local_data, local_label)
  result = abs(temp[0])
  if np.isnan(result):
    result = 0
  return result

