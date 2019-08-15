import numpy as np
import scipy as sp
import scipy.stats

def entropy(feat_vals):
  feat_value_list = list(set(feat_vals))
  feat_value_list = sorted(feat_value_list)
  enable_bin = False
  num_bin = 10
  num_val = len(feat_value_list)
  ten_percentile_list = []
  if (num_val > 100):
    enable_bin = True
    ten_percentile = int(num_val / num_bin)
    for x in range(1, num_bin):
      ten_percentile_list.append(feat_value_list[x * ten_percentile])
    num_val = num_bin

  p_list = np.zeros(num_val)
  if enable_bin:
    for val in feat_vals:
      for i, tpval in enumerate(ten_percentile_list):
        if val <= tpval:
          p_list[i] = p_list[i] + 1
          break
  else:
    for val in feat_vals:
      for i, fval in enumerate(feat_value_list):
        if val <= fval:
          p_list[i] = p_list[i] + 1
          break;
  p_list = p_list / float(num_val)

  ret = sp.stats.entropy(p_list)
  return ret

