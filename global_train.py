import numpy as np
from sklearn.svm import SVC

print(">> Initialize ...")
#Hyper
Vmax = 2
Q = 12
N_nodes = 6
step = 0.125
tau = 0.25  #Equation (34), weight
N_classifiers = 7
N_labels = 2
N_train = 1000
N_test = 50
N_iter = 10

print(">> Load parameters from local nodes ...")
#Get parameters: alphas, local models, train, test dataset
alphas = np.ones((Q, N_nodes, N_classifiers))
gmms = np.array([gmms.OnlineGMM(hyper.std, T=1)] * N_classifiers * N_nodes * Q).reshape(Q, N_nodes, N_classifiers)
#train dataset
Data_train = np.random.normal(0, 1, (N_train, N_classifiers)) #7 samples, 7 attributes
Y_train = np.array((np.mean(Data_train, axis=1) > -0.5), dtype=np.float64)
Y_train[np.squeeze(np.argwhere(Y_train == 0))] -= 1
#Test dataset
Data_test = np.random.normal(0, 1, (N_test, N_classifiers)) #7 samples, 7 attributes
Y_test = np.array(np.mean(Data_test, axis=1) > -0.5, dtype=np.float64)
Y_test[np.squeeze(np.argwhere(Y_test == 0))] -= 1

print(">> Training ....")
for step in range(N_iter):
    #Step 1: Initialization
    #EASY INITIALIZE TWO Q HAVE SAME STATES!
    X = random.normal(loc=0.25, scale=0.5, size=(Q, N_nodes))
    Si = X
    #Step 2: Training SVM
    #Step 2.1: Create vector r
    r = np.zeros((Q, N_nodes))
    for Q_index in range(Q):
        for node_inex in range(N_nodes):    
            r[Q_index][node_inex] = \
                np.sum([alphas[i] * gmms[Q_index][node_inex][i](x) for i in range(N_classifiers)])
    #Step 2.2: Create svm:
    svms = np.array([SVC(kernel = 'linear', C = 1e5)] * Q)
    #Step 2.3: Traing svm:
    for _ in range(Q):
        svms.fit(r, Y_train)
    #Step 3: Update/Select best state
    #Step 4: isTerminate?
    #Step 5: Update state