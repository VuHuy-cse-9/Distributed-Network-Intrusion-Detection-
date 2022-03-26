import matplotlib.pyplot as plt
import utils
import numpy as np

def plot_2ddata(X, y, title="Figure", x_label="x", y_label="y"):
    if X.shape[1] != 2:
        raise Exception("Sorry, plot data can only plot 2-dimension sample")
    
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X {X.shape[0]}different from y {y.shape[0]} ")
    plt.figure()    
    plt.scatter(X[y == 1,0], X[y == 1, 1], c='b')
    plt.scatter(X[y == -1,0], X[y == -1, 1], c='r')
    plt.show()
    
def _norm(x, mean, std):
    return 1.0/(std * np.sqrt(2 * np.pi))*np.exp(-0.5 * ((x - mean) / std) ** 2) 

def plot_norm(mean, std):
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x, _norm(x, mean, std))
    plt.show()
    
def plot_norm_data(X, y, mean, std): 
    if X.shape != (y.shape[0],):
        raise Exception(f"Shape of X should be {(y.shape[0], )}, instead {X.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X {X.shape[0]} different from y {y.shape[0]} ")
    plt.figure()  
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    plt.plot(x, _norm(x, mean, std), c='b')
    plt.plot(X[y == 1], np.zeros((np.sum(y == 1))), c='b', marker='o')
    plt.plot(X[y == -1], np.zeros((np.sum(y == -1))), c='r', marker='o')
    plt.show()
    
def plot_multi_norm_data(X, y, means, stds): 
    """Plot multiple normal distribution graph
    Args:
        X (_array (N_samples, N_features)_): dataset
        y (_N_features_): labels
        means (array_N_features, 2, n_components_): mean for each features
        stds (array_N_features, 2, n_components_): std for each features
    """
    if X.shape != (y.shape[0], means.shape[0]):
        raise Exception(f"Shape of X should be {(y.shape[0], means.shape[0])}, instead {X.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X {X.shape[0]} different from y {y.shape[0]} ")
    
    
    if means.shape[1] != 2:
        raise Exception(f"We only plot for two sample for now ")
    
    if means.shape != stds.shape:
        raise Exception(f"Number of means {means.shape} different from stds {stds.shape} ")
    
    nrows = int(np.ceil(X.shape[1] / 2.0))
    fig, axs = plt.subplots(nrows=nrows, ncols=2) 
    if axs.shape != (nrows, 2):
        axs = axs[None, ]
    for row in range(nrows): 
        for col in range(2):
            if 2*row + col >= X.shape[1]: break; 
            findex = 2*row + col
            axs[row, col].set_title(f"Model for feature's {findex}th")
            for cmp_index in range(means.shape[2]):
                samples1 = np.linspace(means[findex, 0, cmp_index] - 3*stds[findex, 0, cmp_index], means[findex, 0, cmp_index] + 3*stds[findex, 0, cmp_index], 100)
                axs[row, col].plot(samples1, _norm(samples1, means[findex, 0, cmp_index], stds[findex, 0, cmp_index]), c='b')
            axs[row, col].plot(X[y == 1, findex], np.zeros((np.sum(y == 1))), c='b', marker='o')
            for cmp_index in range(means.shape[2]):
                samples2 = np.linspace(means[findex, 1, cmp_index] - 3*stds[findex, 1, cmp_index], means[findex, 1, cmp_index] + 3*stds[findex, 1, cmp_index], 100)
                axs[row, col].plot(samples2, _norm(samples2, means[findex, 1, cmp_index], stds[findex, 1, cmp_index]), c='r')
            axs[row, col].plot(X[y == -1, findex], np.zeros((np.sum(y == -1))), c='r', marker='o')
        
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.show()
    

def plot_one_norm_data(X, y, means, stds): 
    """Plot one normal distribution graph
    Args:
        X (_array (N_samples, N_features)_): dataset
        y (_N_features_): labels
        means (array_N_features, 2, n_components_): mean for each features
        stds (array_N_features, 2, n_components_): std for each features
    """
    if X.shape != (y.shape[0], means.shape[0]):
        raise Exception(f"Shape of X should be {(y.shape[0], means.shape[0])}, instead {X.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X {X.shape[0]} different from y {y.shape[0]} ")
    
    
    if means.shape[1] != 2:
        raise Exception(f"We only plot for two sample for now ")
    
    if means.shape != stds.shape:
        raise Exception(f"Number of means {means.shape} different from stds {stds.shape} ")
    
    fig, axs = plt.subplots(nrows=1, ncols=1) 
   
    findex = 0
    axs.set_title(f"Model for feature's {findex}th")
    for cmp_index in range(means.shape[2]):
        samples1 = np.linspace(means[findex, 0, cmp_index] - 3*stds[findex, 0, cmp_index], means[findex, 0, cmp_index] + 3*stds[findex, 0, cmp_index], 100)
        axs.plot(samples1, _norm(samples1, means[findex, 0, cmp_index], stds[findex, 0, cmp_index]), c='b')
    axs.plot(X[y == 1, findex], np.zeros((np.sum(y == 1))), c='b', marker='o')
    for cmp_index in range(means.shape[2]):
        samples2 = np.linspace(means[findex, 1, cmp_index] - 3*stds[findex, 1, cmp_index], means[findex, 1, cmp_index] + 3*stds[findex, 1, cmp_index], 100)
        axs.plot(samples2, _norm(samples2, means[findex, 1, cmp_index], stds[findex, 1, cmp_index]), c='r')
    axs.plot(X[y == -1, findex], np.zeros((np.sum(y == -1))), c='r', marker='o')
        
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.show()
    
def plot_global_history(history, N_iter, N_states):
    Si_fits = np.array(history["Si_fit"])
    Sg_fits = np.array(history["Sg_fit"])
    Ls = np.array(history["L"])
    dtrs = np.array(history["DETR"])
    fars = np.array(history["FAR"])
    
    if Si_fits.shape != (N_iter, N_states):
        raise Exception(f"Expect Si_fits to be {(N_iter, N_states)}, but have {Si_fits.shape}")
    if Sg_fits.shape != (N_iter, ):
        raise Exception(f"Expect Sg_fits to be {(N_iter, )}, but have {Sg_fits.shape}")
    if Ls.shape != (N_iter, N_states):
        raise Exception(f"Expect Ls to be {(N_iter, N_states)}, but have {Ls.shape}")
    if dtrs.shape != (N_iter, N_states):
        raise Exception(f"Expect dtrs to be {(N_iter, N_states)}, but have {dtrs.shape}")
    if fars.shape != (N_iter, N_states):
        raise Exception(f"Expect fars to be {(N_iter, N_states)}, but have {fars.shape}")

    fig, axs = plt.subplots(nrows=3, ncols=1) 
    axs[0].set_title("Best-fit, Global fit")
    axs[0].plot(np.arange(N_iter), Sg_fits[:], linewidth=2, label="Global fitness")
    for Q_index in range(N_states):
        axs[0].plot(np.arange(N_iter), Si_fits[:, Q_index], linewidth=2, label=f"S{Q_index} fitness")
    axs[0].legend(loc="upper right")  
    
    axs[1].set_title("Contributions of nodes in each states")
    for Q_index in range(N_states):
        axs[1].plot(np.arange(N_iter), Ls[:, Q_index], linewidth=2, label=f"L{Q_index}")
    axs[1].legend(loc="upper right")  
    
    axs[2].set_title("Detection rate, False alarm rate")
    for Q_index in range(N_states):
        axs[2].plot(np.arange(N_iter), dtrs[:, Q_index], linewidth=2, label=f"dtr {Q_index}")
        axs[2].plot(np.arange(N_iter), fars[:, Q_index], linewidth=2, label=f"far {Q_index}")
    axs[2].legend(loc="upper right") 
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    plt.legend()
    plt.show()