import numpy as np
import hyper
import json
import pickle
from models.GMM import OnlineGMM
from models.NewOnlineAdaboost import NewOnlineAdaboost
from utils import convert_json_to_local_models, detection_rate, false_alarm_rate
from DataGenerator.DataGenerator import get_data
from visualize import roc_curve_plot, plot_density_line, plot_multi_norm_data
import argparse

parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
parser.add_argument('--option', default=1, type=int, help='Option for data preprocessing, 0 for category-continous converting, 1 for removing category features.')
args = parser.parse_args()

nodeid = int(args.nodeid)
#====================================================================================
print(">> Loading Adboost combination classifiers ...")
local_model = NewOnlineAdaboost()
with open(f"checkpoint/local/kdd/local_model{nodeid}.json", "rb") as file:
    model_params = json.load(file)
    alphas = model_params["alphas"]
    strong_gmms_list = []
    for featureid in range(hyper.n_features):
        strong_gmms = OnlineGMM(1.0)
        strong_gmms.set_parameters(model_params[f"model_{featureid}"])
        strong_gmms_list.append(strong_gmms)
    strong_gmms_list = np.array(strong_gmms_list)
    local_model.set_params(strong_gmms_list, alphas, hyper.n_features)
    print("========1================")
    print(strong_gmms_list[1].means)
    print("========2================")
    print(strong_gmms_list[9].means)


    
print(">> Loading test dataset")
#====================================================================================
X_test, Y_test, labels_idx, labels, feature_names = get_data(hyper.path_test, hyper.category_features, hyper.skew_features, int(args.option))

print("=====================TEST DATA SUMMARY=========================")
print(f"Number of samples {X_test.shape[0]}, Number of features: {X_test.shape[1]}")
print(f"Train: Number of normal {np.sum(Y_test == 1)}, Number of attack {np.sum(Y_test <= -1)}")
print(f"Number of labels: {np.unique(Y_test).shape[0]}")
print(f"Number of : {len(Y_test[Y_test == -1])}")
print(f"Number of : {len(Y_test[Y_test == -2])}")
print(f"Number of : {len(Y_test[Y_test == -3])}")
print(f"Number of : {len(Y_test[Y_test == -4])}")
print(f"Number of : {len(Y_test[Y_test == -5])}")
print("=================================================================")

Y_roc_test = np.array(np.clip(Y_test, -1, 1), dtype=np.float32)
y_score = local_model.predict_score(X_test)
y_predict = np.sign(y_score)

print(f">> predict right: {np.sum(y_predict == Y_roc_test)}")
print(f">> unique: {np.unique(y_predict)}")
print(f"RESULT SUMMARY:")
tpr = detection_rate(Y_roc_test, y_predict)
fpr = false_alarm_rate(Y_roc_test, y_predict)
print(f">> tpr {tpr * 100}%, fpr {fpr * 100}%")

print(">> ROC curve")
roc_curve_plot(y_test=Y_roc_test, y_score = y_predict)

labels_name = ["normal", "dos", "u2r", "r2l", "prob"]

# strong_gmms = local_model.strong_gmms
strong_gmms = local_model.strong_gmms
means = np.array([strong_gmms[i].means for i in range(hyper.n_features)])
stds = np.array([strong_gmms[i].stds for i in range(hyper.n_features)])
plot_multi_norm_data(X_test, Y_roc_test, means, stds, label_indexs=labels_idx, label_names=labels_name, feature_names=feature_names)

#plot_density_line(y_preds)



    




