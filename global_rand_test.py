import numpy as np
import hyper
import json
import pickle
from models.GMM import OnlineGMM
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from utils import convert_json_to_local_models
from DataGenerator.DataGenerator import load_fake_data
from visualize import roc_curve_plot

#====================================================================================
print(">> Loading Adboost combination classifiers ...")
local_models = [NewOnlineAdaboost()] * hyper.N_nodes
for nodeid in range(hyper.N_nodes):
    #Save model params
    with open(f"checkpoint/global/rand/model{nodeid}.json", "rb") as file:
        model_params = json.load(file)
        alphas = model_params["alphas"]
        strong_gmms = [OnlineGMM(1.0)] * hyper.n_features
        for featureid in range(hyper.n_features):
            strong_gmms[featureid].set_parameters(model_params[f"model_{featureid}"])
        local_models[nodeid].set_params(strong_gmms, alphas, hyper.n_features)

        
print(">> Loading SVM ...")
with open("checkpoint/global/rand/svm.pkl", "rb") as file:
    svm = pickle.load(file)
    
print(">> Loading Particle state ...")
state = None
with open("checkpoint/global/rand/state.json", "rb")  as file:
    state_param = json.load(file)
    state = state_param["state"]
    
global_model = PSOSVMTrainer()
global_model.set_params(svm, state, local_models)
    
print(">> Loading test dataset")
#====================================================================================
N_labels, N_train, N_test = 2, 2000, 500
_, X_test, _, Y_test = load_fake_data(N_train, N_test, hyper.n_features)
N_test = N_test * 2
print(f"Y_test: {Y_test.ravel().shape}")
#Test local models:
print(">> local model: ROC curve")
y_score = [None] * hyper.N_nodes
fpr, tpr = [None] * hyper.N_nodes, [None] * hyper.N_nodes
for i in range(hyper.N_nodes):
    y_score[i] = local_models[i].predict_score(X_test)
    roc_curve_plot(y_test=Y_test, y_score = y_score[i])
    tpr[i], fpr[i] = local_models[i].evaluate(X_test, Y_test)
    
#Test global models:
print(">> Global model: ROC curve")
y_score = global_model.predict(X_test)
roc_curve_plot(y_test=Y_test, y_score = y_score)
gtpr, gfpr = global_model.evaluate(X_test, Y_test)


print(f"RESULT SUMMARY:")
for nodeid in range(hyper.N_nodes):
    print(f">> Node {nodeid}: tpr {tpr[nodeid] * 100}% fpr {fpr[nodeid] * 100}%")
print(f"Global Node: tpr {gtpr * 100}%, fpr {gfpr * 100}%")



    




