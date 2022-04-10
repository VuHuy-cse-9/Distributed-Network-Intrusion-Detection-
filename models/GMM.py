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
            std: (_float_) defautl std
            n_components (_int_): Number of components for each features's GMMs
            T (_float_) threshold used in equation (5)
        """
        self.n_components = n_components
        self.T = T
        self.default_std = std
        self.epsilon = epsilon
        
    def build(self, n_labels):
        """Initialization of GMM components.
        Args:
            n_labels (_int_): _number of labels_
            n_features (_int_): _number of features
        """
        self.n_labels = n_labels
        self.means = np.random.normal(loc=0, scale=5, size=(self.n_labels, self.n_components))
        self.stds = np.ones((n_labels, self.n_components))
        self.weight = np.ones((n_labels, self.n_components))
        self.N = np.ones((self.n_labels, self.n_components), dtype=np.int32)
        self.A = np.ones((self.n_labels, self.n_components))
        
    def fit(self, x, y):
        """Updating Online GMMs for all features
        Args:
            X: (_(float)_): sample
            y: (_int_): label for sample
        Variables:
            Tau (_array-like (n_components_) binary value equation (5)
            Z   (_(float)_) relation value between (x, y) and the GMM equation (8)
            delta (_array-like (n_components)_) probability that (x, y) belong to ith 
                                                                component equation (9)
            t   (_(float))_) indices of GMM's components have min weight
        """
        class_index = self._convert_label_to_index(y)
        # Step 1: Update weight omega_j^y
        Tau = np.squeeze(np.abs((x - self.means[class_index, :]) / self.stds[class_index, :]) < self.T)
        if Tau.shape != (self.n_components, ):
            raise Exception(f"Tau shape should be: {(self.n_components, )}, instead {Tau.shape}")
        
        self.N[class_index] += Tau   
        
        self.weight[class_index] = self.N[class_index] / np.sum(self.N[class_index], keepdims=True)
        
        # Step 2: Calculate the relation between (x, y)
        probs = self.weight[class_index] * self.predict_prob_all(x, class_index) * Tau
        Z = np.sum(probs)
        if Z > 0:      
            # Step 3: Calculate the probability that (x, y) 
            # belongs to the ith component of the GMM.
            delta = probs / Z
            if delta.shape != (self.n_components, ):
                raise Exception(f"delta shape should be: {(self.n_labels, self.n_components)}, instead {delta.shape}")
            
            self.A[class_index] += delta
            
            # Step 4: Update parameters for each components
            self.means[class_index] = \
                    ((self.A[class_index] - delta) * self.means[class_index]  + delta * x) \
                    / self.A[class_index]
            """
            CAUTION: EASILY TO OVERFLOW
            """  
            self.stds[class_index] = np.sqrt(
                ((self.A[class_index] - delta) * (self.stds[class_index] ** 2)) / self.A[class_index]  \
                + ((self.A[class_index] - delta) * delta * (x - self.means[class_index]) ** 2) / self.A[class_index]**2)
        
        else:
            # Step 5: Reset parameters of the weakest components
            t = np.argmin(self.weight[class_index])
            self.N[class_index, t] += 1
            self.A[class_index, t] += 1
            self.means[class_index, t] = x
            self.stds[class_index, t] = self.default_std
            
    def predict_prob_all(self, x, class_index=None):
        if class_index == None: #For Set of sample and non specify class
            stds = self.stds.flatten()[None, :]
            means = self.means.flatten()[None, :]
            # result = -np.log(np.abs(stds)) - np.log(np.sqrt(2 * np.pi)) + \
            #          -0.5 * ((x - means) / stds) ** 2 
            result = 1 / (stds * np.sqrt(2 * np.pi)) * ( np.exp(-0.5 * ((x - means) / stds) ** 2))
            result = result.reshape((x.shape[0],self.n_labels, self.n_components))
            return result
        return 1 / (self.stds[class_index] * np.sqrt(2 * np.pi)) * ( np.exp(-0.5 * ((x - self.means[class_index]) / self.stds[class_index]) ** 2))
        # return -np.log(np.abs(self.stds[class_index])) - np.log(np.sqrt(2 * np.pi)) + \
        #              -0.5 * ((x - self.means[class_index]) / self.stds[class_index]) ** 2 
                    
    def predict_prob(self, x):
        if np.isscalar(x):
            x = np.array([x]).reshape((1, ))
        x = x[:, None]  
        probs = np.sum(self.weight * self.predict_prob_all(x), axis=2)
        return probs
            
    def predict(self, x):
        N_samples = None
        if np.isscalar(x):
            N_samples = 1
        else:
            N_samples = x.shape[0]
        probs = self.predict_prob(x)
        result = np.array(np.min(probs[:, 0][:, None] - (probs[:, 1:]) / (self.n_labels - 1), axis=1) > 0, np.int32) #(Nsamples,)
        result[result == 0] = -1
        return result
        
    def get_parameters(self):
        return {
            "nlabel": self.n_labels,
            "ncomponent": self.n_components,
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "weights": self.weight.tolist(),
        }
        
    def set_parameters(self, model_para):
        self.n_labels = model_para["nlabel"]
        self.n_components = model_para["ncomponent"]
        self.means = np.array(model_para["means"], np.float64)
        self.stds = np.array(model_para["stds"], np.float64)
        self.weight = np.array(model_para["weights"], np.float64)
        
    def get_avg_means(self):
        avg_means = np.mean(self.weight * self.means, axis=1)
        return avg_means

    def get_avg_stds(self):
        avg_stds = np.mean(self.weight * self.stds, axis=1)
        return avg_stds
    
    def _convert_label_to_index(self, label):
        """Convert NIDS'label, which 1 for normal, -1 >= for attack, into index's order.
           with 0 for noraml, 1 <= for attack 
        Args:
            label (array-like (N_samples, )): label for all sample, type string
        """
        if (label > 1) or (label == 0) or ((label * -1) > (self.n_labels - 1)):
            print("Error: label index out of bound")
            return None
        if label == 1:
            return 0
        return label * -1
