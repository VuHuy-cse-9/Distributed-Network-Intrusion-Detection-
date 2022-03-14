import numpy as np
import hyper

class OnlineGMM:
    def __init__(self,
                n_components=1,
                T=0.5):
        """_Initialize parameters_
        
        Args:
            T (_float_) threshold used in equation (5)
        """
        self.n_components = n_components
        self.T = T
        
    def build(self, n_labels = 2):
        """Initialization of GMM components.
        
        Args:
            n_labels (_int_): _number of labels_
        """
        self.n_labels = n_labels
        self.n_gmms = self.n_components * self.n_labels
        self.means = np.zeros((self.n_labels, self.n_components))
        self.stds = np.ones((self.n_labels, self.n_components)) * hyper.std
        self.weight = np.ones((self.n_labels, self.n_components), dtype=float)
        self.N = np.ones((self.n_labels, self.n_components), dtype=np.int8)
        self.A = np.ones((self.n_labels, self.n_components))
    
    def predict_prob(self, x, gmms_indices = None):
        """_return probability of Gaussian_
        
        Args:
            x: (_float_): feature value of sample
            gmss_indices: (_array_like (int, )_): List of selected gmms's indices for predict 
        Variables:
            result: (_array-like (n_labels, n_components)_) probability of each GMM
        """
        if gmms_indices == None:
            return 1.0 / self.stds * np.exp(-0.5 * ((x - self.means) / self.stds) ** 2) 
        return 1.0 / self.stds[gmms_indices, :] * \
            np.exp(-0.5 * ((x - self.means[gmms_indices, :]) / self.stds[gmms_indices, :]) ** 2) 
        
        
    def fit(self, x):
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
        # Step 1: Update weight omega_j^y
        Tau = np.abs((x - self.means) / self.stds) < self.T
        self.N += Tau
        self.weight = self.N / np.sum(self.N, axis=1, keepdims=True)
        
        # Step 2: Calculate the relation between (x, y)
        Z = np.sum(
            self.weight * self.predict_prob(x) * Tau, 
            axis=1
        )
        
        #Only GMMs has Z > 0 go to step 3
        relate_indices = np.nonzero(Z)  #Z > 0
        
        
        if relate_indices[0].size != 0:      
            # Step 3: Calculate the probability that (x, y) 
            # belongs to the ith component of the GMM.
            delta = np.squeeze(
                self.weight[relate_indices, :] * self.predict_prob(x, relate_indices) * Tau[relate_indices, :]
            ) / Z[:, None]
            self.A[relate_indices, :] += delta
            
            # Step 4: Update parameters for each components
            self.means[relate_indices, :] += \
                    ((self.A[relate_indices, :] - delta) * self.means[relate_indices, :]  + delta * x) \
                    / self.A[relate_indices, :]
                  
            """
            CAUTION: EASILY TO OVERFLOW
            """  
            self.stds += np.sqrt(np.squeeze(\
                ((self.A[relate_indices, :] - delta) * (self.stds[relate_indices, :] ** 2)  \
                + (self.A[relate_indices, :] - delta) * delta * (x - self.means[relate_indices, :]) ** 2) \
                / self.A[relate_indices, :]))
        
        # Step 5: Reset parameters of the weakest components
        t = np.argmin(self.weight, axis=1)
        self.N[:, t] += 1
        self.A[:, t] += 1
        self.means[:, t] = x
        self.stds[:, t] = hyper.std
        