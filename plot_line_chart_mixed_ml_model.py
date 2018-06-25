#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import collections
import os

import matplotlib.pyplot as plt


def Plot(name, ylim, x,
         auc_mean, auc_var,
         ig_mean, ig_var,
         linear_mean, linear_var,
         nmi_mean, nmi_var):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(name)
  if ylim is not None:
      ax.set_ylim(ylim[0], ylim[1])
  ax.set_xlabel("Number of Height in Decision Tree")
  ax.set_ylabel(name)
  ax.grid()

  ax.fill_between(x, auc_mean - auc_var,
                  auc_mean + auc_var, alpha=0.1,
                  color="r")
  ax.fill_between(x, ig_mean - ig_var,
                  ig_mean + ig_var, alpha=0.1,
                  color="g")
  ax.fill_between(x, linear_mean - linear_var,
                  linear_mean + linear_var, alpha=0.1,
                  color="b")
  ax.fill_between(x, nmi_mean - nmi_var,
                  nmi_mean + nmi_var, alpha=0.1,
                  color="y")
  ax.plot(x, auc_mean, 'o-', color="r",
          label="AUC")
  ax.plot(x, ig_mean, 'x-', color="g",
          label="IG")
  ax.plot(x, linear_mean, '--', color="b",
          label="Linear")
  ax.plot(x, nmi_mean, '*-', color="y",
          label="NMI")

  ax.legend(loc="best")
  fig.tight_layout()
  fig.savefig(os.getcwd()+'/'+name+'.png', format='png', bbox_inches='tight')
  

def main():

  current_path = os.getcwd()
  df_accuracy = pd.read_csv(current_path+"/Results/Accuracy_05032018.csv")
  df_accuracy_var = pd.read_csv(current_path+"/Results/Accuracy_var_05032018.csv")
  x = df_accuracy["N"].as_matrix()
  name = "Accuracy"
  auc_mean = df_accuracy["DT_AUC"].as_matrix()
  auc_var = df_accuracy_var["DT_AUC"].as_matrix()
  ig_mean = df_accuracy["DT_IG"].as_matrix()
  ig_var = df_accuracy_var["DT_IG"].as_matrix()
  linear_mean = df_accuracy["Linear"].as_matrix()
  linear_var = df_accuracy_var["Linear"].as_matrix()
  nmi_mean = df_accuracy["NMI"].as_matrix()
  nmi_var = df_accuracy_var["NMI"].as_matrix()
  Plot(name, [0.7, 1.0], x,
       auc_mean, auc_var,
       ig_mean, ig_var,
       linear_mean, linear_var,
       nmi_mean, nmi_var)

  df_auc = pd.read_csv(current_path+"/Results/AUC_05032018.csv")
  df_auc_var = pd.read_csv(current_path+"/Results/AUC_var_05032018.csv")
  x = df_auc["N"].as_matrix()
  name = "AUC"
  auc_mean = df_auc["DT_AUC"].as_matrix()
  auc_var = df_auc_var["DT_AUC"].as_matrix()
  ig_mean = df_auc["DT_IG"].as_matrix()
  ig_var = df_auc_var["DT_IG"].as_matrix()
  linear_mean = df_auc["Linear"].as_matrix()
  linear_var = df_auc_var["Linear"].as_matrix()
  nmi_mean = df_auc["NMI"].as_matrix()
  nmi_var = df_auc_var["NMI"].as_matrix()
  Plot(name, [0.4, 0.6], x,
       auc_mean, auc_var,
       ig_mean, ig_var,
       linear_mean, linear_var,
       nmi_mean, nmi_var)

  df_fscore = pd.read_csv(current_path+"/Results/Fscore_05032018.csv")
  df_fscore_var = pd.read_csv(current_path+"/Results/Fscore_var_05032018.csv")
  x = df_fscore["N"].as_matrix()
  name = "Fscore"
  fscore_mean = df_fscore["DT_AUC"].as_matrix()
  fscore_var = df_fscore_var["DT_AUC"].as_matrix()
  ig_mean = df_fscore["DT_IG"].as_matrix()
  ig_var = df_fscore_var["DT_IG"].as_matrix()
  linear_mean = df_fscore["Linear"].as_matrix()
  linear_var = df_fscore_var["Linear"].as_matrix()
  nmi_mean = df_fscore["NMI"].as_matrix()
  nmi_var = df_fscore_var["NMI"].as_matrix()
  Plot(name, [0.7, 1.0], x,
       fscore_mean, fscore_var,
       ig_mean, ig_var,
       linear_mean, linear_var,
       nmi_mean, nmi_var)

if __name__ == '__main__':
  main()

