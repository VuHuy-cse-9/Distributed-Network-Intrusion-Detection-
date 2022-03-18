import numpy as np

class OnlineGMM:
    def __init__(self,
                std,
                n_components=1,
                T=0.5,
                epsilon = 1e-9
                ):
        """_Initialize parameters_
        
        Args:
            T (_float_) threshold used in equation (5)
        """
        self.n_components = n_components
        self.T = T
        self.default_std = std
        self.epsilon = epsilon
        
    def build(self, n_labels = 2):
        """Initialization of GMM components.
        
        Args:
            n_labels (_int_): _number of labels_
        Other:
            Default index 0 is normal classifier
        """
        self.n_labels = n_labels
        self.n_gmms = self.n_components * self.n_labels
        self.means = np.random.normal(loc=0, scale=1, size=(self.n_labels, self.n_components))
        # self.means = np.random.normal(loc=0, scale=1, size=(self.n_labels, self.n_components)) \
        #             * np.arange(n_labels).reshape((n_labels, 1))
        self.stds = np.ones((n_labels, self.n_components))
        self.weight = np.random.normal(loc=0, scale=1, size=(self.n_labels, self.n_components))
        self.N = np.ones((self.n_labels, self.n_components), dtype=np.int32)
        self.A = np.ones((self.n_labels, self.n_components))
    
    def _convert_label_to_index(self, label):
        if (label > 1) or (label == 0) or ((label * -1) > (self.n_labels - 1)):
            print("Error: label index out of bound")
            return None
        if label == 1:
            return 0
        return label * -1
        
    
    def predict_prob(self, x, class_index=None, gmms_indices = None):
        """_return probability of Gaussian_
        Args:
            x: (_float_): feature value of sample
            gmss_indices: (_array_like (int, )_): List of selected gmms's indices for predict 
        Variables:
            result: (_array-like (n_labels, n_components)_) probability of each GMM
        """
        
        if gmms_indices == None:
            return 1.0 / (self.stds[class_index] * np.sqrt(2 * np.pi)) * \
                    np.exp(-0.5 * ((x - self.means[class_index]) / self.stds[class_index]) ** 2) 
        return 1.0 / (self.stds[class_index, gmms_indices] * np.sqrt(2 * np.pi))  * \
            np.exp(-0.5 * ((x - self.means[class_index, gmms_indices]) / self.stds[class_index, gmms_indices]) ** 2) 
        
        
    def fit(self, x, y):
        """Updating Online GMM
        Args:
            x: (_float_): feature value of sample
        Variables:
            Tau (_array-like (nlabels, n_components_) binary value equation (5)
            Z   (_array-like (nlabels, )_) relation value between (x, y) and the GMM equation (8)
            relate_indices (_tuple of array_likes (n < nlabels, )_) indices of GMMS have Z > 0
            delta (_array-like (1, relate_indices, n_components)_) probability that (x, y) belong to ith 
                                                                component equation (9)
            t   (_array_like (nlabels, )_) indices of GMM's components have min weight
        """
        class_index = self._convert_label_to_index(y)
        #print("===========================================================")
        #print(f">> Class index: {class_index}")
        # Step 1: Update weight omega_j^y
        #print(f">> {np.abs((x - self.means) / self.stds)}")
        Tau = np.squeeze(np.abs((x - self.means[class_index, :]) / self.stds[class_index, :]) < self.T)
        self.N += Tau
        #print(f"N: {self.N}")
        self.weight[class_index] = self.N[class_index] / np.sum(self.N[class_index], keepdims=True)
        #print(f"weight: {self.weight}")
        
        # Step 2: Calculate the relation between (x, y)
        Z = np.sum(
            self.weight[class_index] * self.predict_prob(x, class_index) * Tau
        )
        #print(f">> Z: {Z}")
        #print(f"predict_prob: {self.predict_prob(x)}")
        
        #print("-------------------------------------")
        if Z > 0:      
            # Step 3: Calculate the probability that (x, y) 
            # belongs to the ith component of the GMM.
            delta = \
                (self.weight[class_index] * self.predict_prob(x, class_index) * Tau) \
             / Z + self.epsilon
            #print(f">> delta: {delta}")
            self.A[class_index] += delta
            #print(f">> A:  {self.A}")
            
            # Step 4: Update parameters for each components
            self.means[class_index] = \
                    ((self.A[class_index] - delta) * self.means[class_index]  + delta * x) \
                    / self.A[class_index]
            #print(f">> means: {self.means}")
            """
            CAUTION: EASILY TO OVERFLOW
            """  
            self.stds[class_index] = np.sqrt(
                ((self.A[class_index] - delta) * (self.stds[class_index] ** 2)  \
                + (self.A[class_index] - delta) * delta * (x - self.means[class_index]) ** 2) \
                / self.A[class_index])
            #print(f">> stds: {self.stds}")
        
        else:
            # Step 5: Reset parameters of the weakest components
            t = np.argmin(self.weight[class_index])
            self.N[class_index, t] += 1
            self.A[class_index, t] += 1
            self.means[class_index, t] = x
            self.stds[class_index, t] = self.default_std
    
    def predict(self, x):
        """Return predcited class
        Args:
            x (_float_): _feature value_
        return 
        """
        probs = []
        for class_index in range(self.n_labels):
            probs.append(np.sum(self.weight[class_index] * self.predict_prob(x, class_index)))
        probs = np.array(probs)
        #print(probs)
        if (probs[0] > probs[1:] / (self.n_labels - 1)).all(): #How can we know the fist index is labels zeros???
            return 1
        return -1        
        
        