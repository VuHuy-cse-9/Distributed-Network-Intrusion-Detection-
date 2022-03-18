import numpy as np
from sklearn.svm import SVC
from utils import detection_rate, get_models
from GMMs import gmms
from GMMs import hyper

print(">> Initialize ...")
#Hyper
Vmax = 2
Q = 12
N_nodes = 1
step = 0.125
tau = 0.25  #Equation (34), weight
N_classifiers = 7
N_labels = 2
N_train = 1000 #We need huge data
N_test = 50
N_iter = 10
n_components = 4
eta = 1e-5
w = 0.2 #inertia weight
c1, c2 = 0.1, 0.1 #Acceleration constants
u1, u2 = 0.2, 0.3 #Independent random value

print(">> Load parameters from local nodes ...")
#CLONE FROM LOCAL:
#Clone model
alphas = np.ones((Q, N_nodes, N_classifiers))

print(f">> type: {type(gmms.OnlineGMM)}")
model_params = get_models()

local_models = np.empty((Q, N_nodes, N_classifiers), dtype=gmms.OnlineGMM)
for Q_index in range(Q):
    for node_index in range(N_nodes):
        for index in range(N_classifiers):
            local_models[Q_index, node_index, index] = gmms.OnlineGMM(hyper.std, n_components=n_components, T=1)
            local_models[Q_index, node_index, index].set_parameters(model_params[node_index][f"model_{index}"])
            
print("Load model complete")
            
#Clone train dataset
Data_train = np.random.normal(0, 1, (N_train, N_classifiers)) #7 samples, 7 attributes
Y_train = np.array((np.mean(Data_train, axis=1) > -0.5), dtype=np.float64)
Y_train[np.squeeze(np.argwhere(Y_train == 0))] -= 1

#Clone test dataset
Data_test = np.random.normal(0, 1, (N_test, N_classifiers)) #7 samples, 7 attributes
Y_test = np.array(np.mean(Data_test, axis=1) > -0.5, dtype=np.float64)
Y_test[np.squeeze(np.argwhere(Y_test == 0))] -= 1


#VARIABLES:
Si = np.zeros((Q, N_nodes))
Si_fit = None
Sg = [N_nodes,]
Vi = np.zeros((Q, N_nodes))

#INITIALIZE MODELS:
svcs = []
for Q_index in range(Q):
    svcs.append(SVC(kernel = 'linear', C = 1e5))

print(f">> gmms: {local_models.shape}, type: {type(local_models[0, 0, 0])}")
#HELP FUNCTION
def data_to_vector(Data, X):
    r = np.zeros((Q, N_nodes, Data.shape[0]))
    for Q_index in range(Q):
        for node_index in np.nonzero(X[Q_index]>0)[0]:    
            r[Q_index][node_index] = \
                X[Q_index, node_index] * \
                np.sum([alphas[Q_index][node_index][i] * \
                    local_models[Q_index, node_index, i].predict(Data[i]) \
                        for i in range(N_classifiers)])
    return np.transpose(r, (0, 2, 1))

def fitness_function(dtr, tau, L):
    return tau * dtr + (1 - tau) * np.log(np.maximum((N_nodes - L) / (N_nodes * 1.0), eta))

def constraint_velocity(V):
    """ Limit the range of value of V
    Args:
        V (_Q, Nodes_): _Velocities for eachs states_
    """
    #TODO: Add constrainted
    return V
    

print(">> Training ....")
#Initialize state
#BUG: TWO Q MAY HAVE SAME STATES!
X = np.random.normal(loc=0.25, scale=0.5, size=(Q, N_nodes))
X = X  * (X > 0)
Si = X  

for step in range(N_iter):
    print(f"=============STEP {step + 1}===============")
    #Get number of nodes be used in particle i
    L = np.sum(X > 0, axis=1)
    
    #Training SVM
    r_train = data_to_vector(Data_train, X)
    for Q_index in range(Q):
        svcs[Q_index].fit(r_train[Q_index], Y_train)
        
    #Evaluate svm:
    r_test = data_to_vector(Data_test, X)
    dtr = np.zeros((Q, ))
    for Q_index in range(Q):
        predicts = svcs[Q_index].predict(r_test[Q_index])
        # print(f">> predict: {predicts.shape}")
        # print(f">> Y_test: {Y_test.shape}")
        # print(f">> Detection rate: {detection_rate( Y_test, svcs[Q_index].predict(r_test[Q_index]))}")
        dtr[Q_index] =  detection_rate( 
                            Y_test, 
                            svcs[Q_index].predict(r_test[Q_index])
                        )
    
    #Calculate fitness & Update best state
    X_fit = fitness_function(dtr, tau=tau, L=L)
    if Si_fit is None:
        Si_fit = X_fit
    else:
        for Q_index in range(Q):
            if X_fit[Q_index] > Si_fit[Q_index]:
                Si[Q_index] = X[Q_index]
                Si_fit[Q_index] = X[Q_index]
                
    #Update global state:
    Sg = Si[np.argmax(Si_fit)]
    
    for Q_index in range(Q):
        #Calculate velocities
        Vi[Q_index] = constraint_velocity(
            w * Vi[Q_index] + c1 * u1 * (Si[Q_index] - X[Q_index]) + c2 * u2 * (Sg - X[Q_index])
        )
        #Evolve particle state
        X[Q_index] += Vi[Q_index]
    