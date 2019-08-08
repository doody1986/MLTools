import TreeEnsembleEval
import TreeEnsembleFeatureSelection
import os
import sys
import datetime
import pandas as pd

# Evaluate the performance given different ensemble number across all features
selected_features = []
label_name = "PPTERM"
num_ensembles = [11, 31, 51, 71, 91]

cur_working_path = os.getcwd()

today = datetime.datetime.now().strftime("%m%d%Y")

extension = ".csv"

completeness_ratios = [80]

data_pathes = []
for ratio in completeness_ratios:
  data_v1_v2 = "data_preprocessing/protect_data_no_prefill_"+str(ratio)+"_V1_V2.csv"
  data_v1_v2_v3 = "data_preprocessing/protect_data_no_prefill_"+str(ratio)+"_V1_V2_V3.csv"
  data_v1_v2_v3_v4 = "data_preprocessing/protect_data_no_prefill_"+str(ratio)+"_V1_V2_V3_V4.csv"
  data_pathes.append(os.path.join(cur_working_path, data_v1_v2))
  data_pathes.append(os.path.join(cur_working_path, data_v1_v2_v3))
  data_pathes.append(os.path.join(cur_working_path, data_v1_v2_v3_v4))

evaluate_ensemble_size = False
evaluate_feature_selection_method = True

num_rounds = 20
df1 = pd.DataFrame(columns=['Accuracy', 'AUC', 'Ensemble Size', 'Data File'])
exp1_filename = "no_prefill_performance_with_different_ensemble_size_"+today+extension
if evaluate_ensemble_size:
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

# method_options = ['CLA', 'WMA', 'OFA', 'CAA', 'MAA']
method_options = ['MAA']
missing_rate_table_path = "data_preprocessing/missing_rate_table.csv"
missing_rate_table = pd.read_csv(missing_rate_table_path)
num_ensemble_by_data = [91, 91, 91]
num_selected_features = 20

df2 = pd.DataFrame(columns=['Accuracy', 'AUC', 'Method', 'Data File'])
# exp2_filename = "performance_with_different_method_"+str(num_selected_features)+"features.csv"
exp2_filename = "no_prefill_performance_with_different_methods_"+today+extension
if evaluate_feature_selection_method:
  for i in range(len(data_pathes)):
    print("\nIn "+data_pathes[i])
    input_data = pd.read_csv(data_pathes[i])
    ensemble_size = num_ensemble_by_data[i%len(num_ensemble_by_data)]
    for method in method_options:
      print("\n\tFeature selection method: " + method)
      selected_features = TreeEnsembleFeatureSelection.Run(input_data, label_name,
                                                           ensemble_size, method,
                                                           num_selected_features,
                                                           missing_rate_table)
      print("\tSelected features: ")
      print(selected_features)
      missing_rate = 0.0
      for sfeat in selected_features:
        missing_rate += TreeEnsembleFeatureSelection.MissingRate(missing_rate_table, sfeat,
                                                                 input_data.columns.to_list())
      missing_rate /= float(num_selected_features)
      print("Averaged missing rate of selected features: "+str(missing_rate))
      final_accuracy = 0.0
      final_auc = 0.0
      for j in range(num_rounds):
        # Print the progress
        sys.stdout.write('\r>> Evaluate Progress %.1f%%' % (float(j + 1) / float(num_rounds) * 100.0))
        sys.stdout.flush()
        accuracy, fnr, fpr, auc = TreeEnsembleEval.Run(input_data, label_name, ensemble_size, selected_features)
        final_accuracy += accuracy
        final_auc += auc
      final_accuracy /= float(num_rounds)
      final_auc /= float(num_rounds)
      temp = pd.DataFrame([[final_accuracy, final_auc, method, data_pathes[i]]], columns=df2.columns)
      df2 = df2.append(temp, ignore_index=True)
  df2.to_csv(exp2_filename, index=False)


