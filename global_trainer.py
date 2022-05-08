from utils import detection_rate, false_alarm_rate, send_model, clone_model_from_local
import hyper
from visualize import plot_global_history, roc_curve_plot
from models.NewOnlineAdaboost import NewOnlineAdaboost
from models.PSOSVM import PSOSVMTrainer
from DataGenerator.DataGenerator import get_local_train_dataset, get_part_local_train_dataset, get_local_test_dataset

import numpy as np
import json
from tqdm import tqdm
import argparse
from sklearn.svm import SVC

import os.path

import pickle

class GlobalTrainer:
    def __init__(self, options):
        self.opt = options
        #Load local train dataset:
        self.data={}
        if self.opt.train_mode == "full":
            self.data["X_train"], self.data["Y_train"], \
                self.data["labels_idx"], self.data["labels"], self.data["feature_names"] = \
                get_local_train_dataset(self.opt.nodeid, 
                                        self.opt.path_train, 
                                        self.opt.log_transform,
                                        self.opt.cate2cont)
        elif self.opt.train_mode == "part":
            self.data["X_train"], self.data["Y_train"], \
                self.data["labels_idx"], self.data["labels"], self.data["feature_names"] = \
                    get_part_local_train_dataset(self.opt.nodeid, 
                                                self.opt.path_train, 
                                                self.opt.log_transform,
                                                self.opt.cate2cont)
        elif self.opt.train_mode == "random":
            raise Exception("Feature haven't implemented!")
        else:
            raise Exception("Invalid train mode")
        
        if self.opt.cate2cont == "integer":
            self.n_features = 41
            self.n_labels = 5
        elif self.opt.cate2cont == "remove":
            self.n_features = 32
            self.n_labels = 5
        
        #Load global train dataset:
        self.data['X_global_train'], self.data['Y_global_train'] = self._select_global_data(self.data["X_train"], self.data["Y_train"], self.n_labels - 1, self.opt.N_global_train)
        self.data['X_global_test'], self.data['Y_global_test'] = self._select_global_data(self.data["X_train"], self.data["Y_train"], self.n_labels - 1, self.opt.N_global_test)
        
        #Get test dataset:
        self.data["X_test"], self.data["Y_test"], _, _, _ = \
                get_local_test_dataset(
                        self.opt.nodeid,    
                        self.opt.path_test, 
                        self.opt.log_transform,
                        self.opt.cate2cont)
        
        #Prepare model:
        self.model = NewOnlineAdaboost(
            self.opt.r,
            self.opt.p,
            self.opt.P,
            self.opt.gamma,
            self.opt.beta,
            self.opt.n_components,
            self.opt.T,
            self.opt.default_std
        )   
        self.model.build(n_labels=self.n_labels, n_features=self.n_features)
        
        self.global_trainer = PSOSVMTrainer(
            N_states=self.opt.N_particles,
            N_nodes=self.opt.N_nodes,
            N_iter=self.opt.N_iter,
            tau=self.opt.tau,
            n_features=self.n_features,
            inertia_weight_mode=self.opt.inertia_weight_mode,
            c1=self.opt.c1,
            c2=self.opt.c2,
            u1=self.opt.u1,
            u2=self.opt.u2,
            V_max=self.opt.V_max
        )
        
    def train_local(self):
        self.model.fit(self.data["X_train"], self.data["Y_train"])
        
    def _send_model(self):
        model_dict = self.model.to_dict()
        model_dict["node"] = self.opt.nodeid
        send_model(model_dict, self.opt.kafka_server)
        
    def receive_model(self):
        strong_gmms_list, global_alphas = clone_model_from_local(
                                            curr_nodeid=self.opt.nodeid, 
                                            N_nodes=self.opt.N_nodes, 
                                            N_classifiers=self.n_features,
                                            n_components=self.opt.n_components,
                                            url=self.opt.kafka_server
                                        )
        strong_gmms_list[self.opt.nodeid] = self.model.strong_gmms
        global_alphas[self.opt.nodeid] = self.model.alphas
        local_models = []
        for i in range(self.opt.N_nodes):
            local_model = NewOnlineAdaboost(
                            self.opt.r,
                            self.opt.p,
                            self.opt.P,
                            self.opt.gamma,
                            self.opt.beta,
                            self.opt.n_components,
                            self.opt.T,
                            self.opt.default_std
                        )
            local_model.set_params(strong_gmms_list[i], global_alphas[i], self.n_features)
            local_models.append(local_model)
        local_models = np.array(local_models)
        self.global_trainer.build(local_models)

    def train_global(self):
        self.history = self.global_trainer.fit(
                        self.data['X_global_train'], 
                        self.data['Y_global_train'], 
                        self.data['X_global_test'], 
                        self.data['Y_global_test'])
        
    def eval_global(self):
        if not os.path.exists(self.opt.path_result + f"model{self.opt.nodeid}/"):
            os.mkdir(self.opt.path_result + f"model{self.opt.nodeid}/")
        #Test global models:
        Y_roc_test = np.array(np.clip(self.data['Y_test'], -1, 1), dtype=np.float32)
        y_pred = self.global_trainer.predict(self.data['X_test'])

        print(f"======PREDICT RESULT SUMMARY========")
        print(f"shape of result: {np.array(y_pred).shape}")
        print(f"Label that predict: {np.unique(y_pred)}")
        print(f"Number that match: {np.sum(y_pred == Y_roc_test)}")

        print("========RESULT============")
        dtr = detection_rate(Y_roc_test, y_pred)
        far = false_alarm_rate(Y_roc_test, y_pred)
        print(f"dtr: {dtr * 100}%, far: {far * 100}")

        roc_curve_plot(y_test=Y_roc_test, 
                       y_score = y_pred, 
                       path_dir=self.opt.path_result + f"model{self.opt.nodeid}/")
        
        plot_global_history(
            self.history, 
            self.opt.N_iter, 
            self.opt.N_particles,
            path_dir=self.opt.path_result + f"model{self.opt.nodeid}/")
    
    def _select_global_data(self, X, Y, n_attack, n_samples):
        shuffle_index = self._get_random_index_array(X[Y == 1].shape[0])
        X_select = X[Y == 1][shuffle_index][:n_samples]
        Y_select = Y[Y == 1][shuffle_index][:n_samples]
        for i in range(n_attack):
            attack_index = -1*i - 1
            shuffle_index = self._get_random_index_array(X[Y == attack_index].shape[0])
            X_attack = X[Y == attack_index][shuffle_index][:n_samples]
            Y_attack = Y[Y == attack_index][shuffle_index][:n_samples]
            X_select = np.concatenate((X_select, X_attack), axis=0)
            Y_select = np.concatenate((Y_select, Y_attack), axis=0)
        shuffle_index = self._get_random_index_array(X_select.shape[0])
        X_select = X_select[shuffle_index]
        Y_select = Y_select[shuffle_index]
        return X_select, Y_select
    
    def _get_random_index_array(self, N):
        array_index = np.arange(N)
        np.random.shuffle(array_index)
        return array_index
    
    def save(self):
        state = self.global_trainer.Sg
        svc = self.global_trainer.global_svc
        state_dict =  {"state": state.tolist()}

        if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/"):
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "state.json", "w") as outfile:
                outfile.write(json.dumps(state_dict))
        else:
            os.mkdir(self.opt.path_save_dir + f"model{self.opt.nodeid}")
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "state.json", "x") as outfile:
                outfile.write(json.dumps(state_dict))
                
        if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "history.json"):
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "history.json", "w") as outfile:
                outfile.write(json.dumps(self.history))
        else:
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "history.json", "x") as outfile:
                outfile.write(json.dumps(self.history))
            
        # save
        if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "svm.pkl"):
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "svm.pkl", "wb") as outfile:
                pickle.dump(svc,outfile)
        else:
            with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "svm.pkl", "xb") as outfile:
                pickle.dump(svc,outfile)

        if not os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "local_models/"):
            os.mkdir(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "local_models/")
            
        local_models = self.global_trainer.local_models
        for i, local_model in enumerate(local_models):
            #Save model params
            model_dict = local_model.to_dict()
            model_dict["node"] = i
            params = json.dumps(model_dict)
            if os.path.exists(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "local_models/" + f"model{i}.json"):
                with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "local_models/" + f"model{i}.json", "w") as outfile:
                    outfile.write(params)
            else:
                with open(self.opt.path_save_dir + f"model{self.opt.nodeid}/" + "local_models/" + f"model{i}.json", "x") as outfile:
                    outfile.write(params)
        
    def summary_dataset(self):
        #Summerize data:
        print("=====================LOCAL DATA SUMMARY=========================")
        print(f"Train: Number of normal {np.sum(self.data['Y_train'] == 1)}, Number of attack {np.sum(self.data['Y_train'] <= -1)}")
        print(f"Number of features: {self.data['X_train'].shape[1]}")
        print(f"normal: {self.data['Y_train'][self.data['Y_train'] == 1].shape[0]}")
        print(f"neptune: {self.data['Y_train'][self.data['Y_train'] == hyper.attack_global_train['neptune.']].shape[0]}")
        print(f"snmpgetattack: {self.data['Y_train'][self.data['Y_train'] == hyper.attack_global_train['snmpgetattack.']].shape[0]}")
        print(f"mailbomb: {self.data['Y_train'][self.data['Y_train'] == hyper.attack_global_train['mailbomb.']].shape[0]}")
        print(f"smurf: {self.data['Y_train'][self.data['Y_train'] == hyper.attack_global_train['smurf.']].shape[0]}")
        print("=====================GLOBAL TRAIN DATA SUMMARY=========================")
        print(f"Number of train samples {self.data['X_global_train'].shape[0]}, Number of features: {self.data['X_global_train'].shape[1]}")
        print(f"Number of train normal {np.sum(self.data['Y_global_train'] == 1)}, Number of attack {np.sum(self.data['Y_global_train'] <= -1)}")
        print(f"Number of labels: {np.unique(self.data['Y_global_train']).shape[0]}")
        print(f"Number of : {len(self.data['Y_global_train'][self.data['Y_global_train'] == -1])}")
        print(f"Number of : {len(self.data['Y_global_train'][self.data['Y_global_train'] == -2])}")
        print(f"Number of : {len(self.data['Y_global_train'][self.data['Y_global_train'] == -3])}")
        print(f"Number of : {len(self.data['Y_global_train'][self.data['Y_global_train'] == -4])}")
        print(f"Number of : {len(self.data['Y_global_train'][self.data['Y_global_train'] == -5])}")
        print("---------------------GLOBAL TEST DATA SUMMARY----------------------------------------")
        print(f"Number of test samples {self.data['X_global_test'].shape[0]}, Number of features: {self.data['X_global_test'].shape[1]}")
        print(f"Number of test normal {np.sum(self.data['Y_global_test'] == 1)}, Number of attack {np.sum(self.data['Y_global_test'] <= -1)}")
        print(f"Number of labels: {np.unique(self.data['Y_global_test']).shape[0]}")
        print(f"Number of : {len(self.data['Y_global_test'][self.data['Y_global_test'] == -1])}")
        print(f"Number of : {len(self.data['Y_global_test'][self.data['Y_global_test'] == -2])}")
        print(f"Number of : {len(self.data['Y_global_test'][self.data['Y_global_test'] == -3])}")
        print(f"Number of : {len(self.data['Y_global_test'][self.data['Y_global_test'] == -4])}")
        print(f"Number of : {len(self.data['Y_global_test'][self.data['Y_global_test'] == -5])}")
        print("=====================TEST DATA SUMMARY=========================")
        print(f"Number of samples {self.data['X_test'].shape[0]}, Number of features: {self.data['X_test'].shape[1]}")
        print(f"Number of normal {np.sum(self.data['Y_test'] == 1)}, Number of attack {np.sum(self.data['Y_test'] <= -1)}")
        print(f"Number of labels: {np.unique(self.data['Y_test']).shape[0]}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -1])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -2])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -3])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -4])}")
        print(f"Number of : {len(self.data['Y_test'][self.data['Y_test'] == -5])}")
        print("=================================================================")