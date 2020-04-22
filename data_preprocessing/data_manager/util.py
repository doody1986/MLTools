import numpy as np
import scipy as sp
import scipy.stats
import math

def plist(feat_vals):
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

  if enable_bin:
    p_list = np.zeros(num_bin)
  else:
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

  return p_list

def entropy(feat_vals):
  p_list = plist(feat_vals)
  ret = sp.stats.entropy(p_list)
  return ret

def NormMI(feat_vals_before_process, feat_vals_after_process, log_base):
  p_list_x = plist(feat_vals_before_process)
  p_list_y = plist(feat_vals_after_process)

  mi_xy = 0.0
  # For each random
  for px in p_list_x:
    for py in p_list_y:
      pxy = px * py
      print("px",px)
      print("py",py)
      print("pxy",pxy)
      if pxy != 0.0:
        mi_xy += pxy * math.log((pxy / (px * py)), log_base)
        print("mi_xy", mi_xy)
  exit()
  print("mi_xy:", mi_xy)

  mi_x = 0
  for px in p_list_x:
    for py in p_list_x:
      pxy = px * py
      if pxy != 0.0:
        mi_x += pxy * math.log((pxy / (px * py)), log_base)
  print("mi_x:", mi_x)
  mi_y = 0
  for px in p_list_y:
    for py in p_list_y:
      pxy = px * py
      if pxy != 0:
        mi_y += pxy * math.log((pxy / (px * py)), log_base)
  print("mi_y:", mi_y)
  exit()

  nmi = mi_xy / max(mi_x, mi_y)

  return nmi

