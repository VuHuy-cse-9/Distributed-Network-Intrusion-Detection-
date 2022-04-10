import numpy as np
from sklearn.svm import SVC
from utils import clone_model_from_local
import hyper
from visualize import plot_global_history
from tqdm import tqdm


class PSOSVMTrainer():
    def __init__(self):
        self.Vmax = hyper.V_max
        self.Q = hyper.N_states
        self.N_nodes = hyper.N_nodes
        self.tau = hyper.tau
        self.N_classifiers = hyper.n_features
        self.N_iter = hyper.N_iter
        self.eta = 1e-5
        self.w = hyper.w                        #inertia weight
        self.c1, self.c2 = hyper.c1, hyper.c2
        self.u1, self.u2 = hyper.u1, hyper.u2
        
    def build(self, local_models):
        """_summary_

        Args:
            local_models (_array-like OnlineAdaboost (N_nodes, ))
        """
        self.Sg = np.zeros([self.N_nodes,])         #Best global state of every particles
        self.global_svc = None
        self.Si = None                           #Best state of particles, equation (35)
        self.Si_fit = None                          #Fitness value of best state of particles
        self.svcs = []
        for Q_index in range(self.Q):
            self.svcs.append(SVC(kernel = 'rbf', C = 1e5))#Every states have a private support vector machine
        self.local_models = local_models
        
    def data_to_vector(self, Data, X_state):
        """ Transform raw data into feature vector using Local models
        Args:
            Data (array_(N_samples, N_features)): dataset
        Returns:
            r: (array (N_samples, self.N_nodes)), feature vector
        """
        rs = []
        for x_state in X_state:
            selected_node_indexs = np.squeeze(np.argwhere(x_state > 0))
            L = len(selected_node_indexs)
            r = np.zeros((L, Data.shape[0]))
            for index, node_index in enumerate(selected_node_indexs):
                r[index] = self.local_models[node_index].predict_score(Data)
            r = np.transpose(r, (1, 0))
            rs.append(r)
        return rs

    def fitness_function(self, dtr, L):
        """Calculate fitness value using equation (34)

        Args:
            dtr (array (N_states, )): detection rate by private svm on test dataset.
            L (array (N_states,): Number of nodes be selected on each state, equation (34)

        Returns:
            : (array, (N_states,)): fitness value for each state
        """
        return self.tau * dtr + (1 - self.tau) * np.log(np.maximum((self.N_nodes - L) / (self.N_nodes * 1.0), self.eta))

    def constraint_velocity(self, V):
        """ Limit the range of value of V
        Args:
            V (_Q, Nodes_): _Velocities for eachs states, equation (38)
        """
        #TODO: Add constrainted
        return V
    
    def fit(self, X_train, Y_train, X_val, Y_val):
        history = {
            "Si_fit": [],                           #Best fit of particle, (N_iter, N_states)
            "Sg_fit": [],                           #Global fit, (N_iter, )
            "L": [],                                #Particle state, (N_iter, N_states)
            "DETR": [],                             #Detection rate (N_iter, N_states)
            "FAR": []                               #False alarm rate (N_iter, N_states)
        }
        #BUG: TWO self.Q MAY HAVE SAME STATES!  
        X_state = np.clip(np.random.normal(loc=0.25, scale=0.5, size=(self.Q, self.N_nodes)), 1e-5, 1.0)
        X_state = X_state  * (X_state > 0)
        self.Si = X_state
        Vi = np.zeros((self.Q, self.N_nodes))  #Velocities of each particles, equation (37)
        
        for step in tqdm(range(self.N_iter)):
            #MODIFY: L(int) to L(weight) in (34)
            L_weight = np.sum(X_state, axis=1)

            history["L"].append(L_weight.tolist())
            
            if L_weight.shape != (self.Q,):
                raise Exception(f"Expect L_weight has shape {(self.Q, )}, instead {L_weight.shape}") 
            
            #Training SVM
            r_train = self.data_to_vector(X_train, X_state)
            
            for Q_index in range(self.Q):
                self.svcs[Q_index].fit(r_train[Q_index], Y_train)    
            
            #Evaluate svm:
            r_test = self.data_to_vector(X_val, X_state)
            
            dtr = np.zeros((self.Q, ))  #Detection rate
            far = np.zeros((self.Q, ))  #False alarm rate
            for Q_index in range(self.Q):
                predicts = self.svcs[Q_index].predict(r_test[Q_index])
                dtr[Q_index] =  detection_rate( 
                                    Y_val, 
                                    predicts
                                )
                far[Q_index] = false_alarm_rate(
                                    Y_val, 
                                    predicts
                                )       
            
            history["DETR"].append(dtr.tolist())
            history["FAR"].append(far.tolist())
            #Calculate fitness & Update best state
            X_fit = self.fitness_function(dtr, L=L_weight)
            if X_fit.shape != (self.Q,):
                raise Exception(f"Expect X_fit has shape {(self.Q,)}, instead {X_fit.shape}")
            
            if self.Si_fit is None:
                self.Si_fit = X_fit
            else:
                for Q_index in range(self.Q):
                    if X_fit[Q_index] > self.Si_fit[Q_index]:
                        self.Si[Q_index] = X_state[Q_index]
                        self.Si_fit[Q_index] = X_fit[Q_index]
            if self.Si.shape != (self.Q,self.N_nodes):
                raise Exception(f"Expect Si has shape {(self.Q, self.N_nodes)}, instead {self.Si.shape}")
                        
            #Update global state:
            idx = np.argmax(self.Si_fit)
            self.Sg = self.Si[idx]
            self.global_svc = self.svcs[idx]
            Sg_fit = self.Si_fit[idx]
            
            if self.Sg.shape != (self.N_nodes,):
                raise Exception(f"Expect Sg has shape {(self.N_nodes,)}, instead {self.Sg.shape}")
            
            history["Si_fit"].append(self.Si_fit.tolist())
            history["Sg_fit"].append(Sg_fit)
            
            for Q_index in range(self.Q):
                #Calculate velocities
                Vi[Q_index] = self.constraint_velocity(
                    self.w * Vi[Q_index] + self.c1 * self.u1 * (self.Si[Q_index] - X_state[Q_index]) + self.c2 * self.u2 * (self.Sg - X_state[Q_index])
                )
                #Evolve particle state
                X_state[Q_index] += Vi[Q_index]
        return history

    def set_params(self, svc, global_state, local_models):
        """_summary_

        Args:
            svc (_slklearn svm_): trained support vector machine 
            global_state (_array-like (N_classifier, )_): global particle state
            local_models (_array-like (N_nodes, )_): Online Adaboost for all nodes
        """
        self.global_svc = svc
        self.Sg = np.array(global_state)
        self.local_models = local_models
    
    def predict(self, X_test):
        selected_node_indexs = np.squeeze(np.argwhere(self.Sg > 0))
        L = len(selected_node_indexs)
        r = np.zeros((L, X_test.shape[0]))
        for index, node_index in enumerate(selected_node_indexs):
            r[index] = self.local_models[node_index].predict_score(X_test)
        r = np.transpose(r, (1, 0))
        y = self.global_svc.predict(r)
        return y