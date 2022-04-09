import numpy as np
import hyper
import json
import pickle
from models.GMM import OnlineGMM
from models.NewOnlineAdaboost import NewOnlineAdaboost
from utils import convert_json_to_local_models
from DataGenerator.DataGenerator import load_fake_data
from visualize import roc_curve_plot, plot_density_line
import argparse

parser = argparse.ArgumentParser(description='Distributed Intrusion Detection')
parser.add_argument('--nodeid', default=1.0, type=int, help='Node"s id')
args = parser.parse_args()

nodeid = int(args.nodeid)
#====================================================================================
print(">> Loading Adboost combination classifiers ...")
local_model = NewOnlineAdaboost()
with open(f"checkpoint/local/rand/local_model{nodeid}.json", "rb") as file:
    model_params = json.load(file)
    alphas = model_params["alphas"]
    strong_gmms_list = []
    for featureid in range(hyper.n_features):
        strong_gmms = OnlineGMM(1.0)
        strong_gmms.set_parameters(model_params[f"model_{featureid}"])
        strong_gmms_list.append(strong_gmms)
    strong_gmms_list = np.array(strong_gmms_list)
    local_model.set_params(strong_gmms_list, alphas, hyper.n_features)

    
print(">> Loading test dataset")
#====================================================================================
N_labels, N_train, N_test = 2, 2000, 100
_, X_test, _, Y_test = load_fake_data(N_train, N_test, hyper.n_features)
N_test = N_test * 2
print(f"Y_test: {Y_test.ravel().shape}")
#Test local models:

print(">> local model: ROC curve")
y_score = local_model.predict_score(X_test)
y_preds = np.sign(y_score)
print(f">> y score: {y_score}")
print(f">> y test: {Y_test}")
roc_curve_plot(y_test=Y_test, y_score = y_score)
tpr, fpr = local_model.evaluate(X_test, Y_test)


print(f"RESULT SUMMARY:")
print(f">> tpr {tpr * 100}%, fpr {fpr * 100}%")

#plot_density_line(y_preds)



    




