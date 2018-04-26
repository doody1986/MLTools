import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

def Calc(data, label):
  null_flags = np.isnan(data).tolist()
  valid_idx = [idx for idx, value in enumerate(null_flags) if value == False]
  local_data = data[[valid_idx]]
  local_label = label[[valid_idx]]
  result = normalized_mutual_info_score(local_data, local_label)
  return result

