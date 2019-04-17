import TreeEnsembleEval
import os
import pandas as pd

# Evaluate the performance given different ensemble number across all features
selected_features = []
label_name = "PPTERM"
num_ensembles = [11, 31, 51, 71, 91]

cur_working_path = os.getcwd()

data_v1_v2 = "protect_data_V1_V2.csv"
data_v1_v2_v3 = "protect_data_V1_V2_V3.csv"
data_v1_v2_v3_v4 = "protect_data_V1_V2_V3_V4.csv"

data_pathes = []
data_pathes.append(os.path.join(cur_working_path, data_v1_v2))
data_pathes.append(os.path.join(cur_working_path, data_v1_v2_v3))
data_pathes.append(os.path.join(cur_working_path, data_v1_v2_v3_v4))

num_rounds = 10
df1 = pd.DataFrame(columns=['Accuracy', 'AUC', 'Ensemble Size', 'Data File'])
exp1_filename = "performance_with_various_ensemble_size.csv"
for data_path in data_pathes:
  input_data = pd.read_csv(data_path)
  for num in num_ensembles:
    final_accuracy = 0.0
    final_auc = 0.0
    for i in range(num_rounds):
      accuracy, fnr, fpr, auc = TreeEnsembleEval.Run(input_data, label_name, num)
      final_accuracy += accuracy
      final_auc += auc
    final_accuracy /= float(num_rounds)
    final_auc /= float(num_rounds)
    temp = pd.DataFrame([[final_accuracy, final_auc, num, data_path]], columns=df1.columns)
    df1 = df1.append(temp, ignore_index=True)
df1.to_csv(exp1_filename, index=False)
