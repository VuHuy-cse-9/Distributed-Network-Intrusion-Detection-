import matplotlib.pyplot as plt
import utils
import numpy as np
import hyper 
from sklearn.metrics import roc_curve, auc
    
def _norm(x, mean, std):
    return 1.0/(std * np.sqrt(2 * np.pi))*np.exp(-0.5 * ((x - mean) / std) ** 2) 
def plot_multi_norm_data(X, 
                         y, 
                         means, 
                         stds, 
                         label_indexs, 
                         label_names, 
                         feature_names, 
                         path_dir,
                         n_components): 
    """Plot multiple normal distribution graph
    Args:
        X (_array (N_samples, N_features)_): dataset
        y (_N_samples_): labels
        means (array_N_features, 2, n_components_): mean for each features
        stds (array_N_features, 2, n_components_): std for each features
    """
    if X.shape != (y.shape[0], means.shape[0]):
        raise Exception(f"Shape of X should be {(y.shape[0], means.shape[0])}, instead {X.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise Exception(f"Number of sample in X {X.shape[0]} different from y {y.shape[0]} ")
    
    if means.shape != stds.shape:
        raise Exception(f"Number of means {means.shape} different from stds {stds.shape} ")
    
    color = ['b', 'r', 'c', 'm', 'yellow', 'g','black']
    labels_index = np.unique(label_indexs)[::-1]
    for idx in range(0, X.shape[1], 4):
        nrows = 4
        fig, axs = plt.subplots(nrows=nrows, ncols=2) 
        if axs.shape != (nrows, 2):
            axs = axs[None, ]
        for row in range(nrows): 
            for col in range(2):
                if (2*row + col + idx) >= X.shape[1]: break; 
                findex = 2*row + col + idx
                axs[row, col].set_title(f"{feature_names[findex]}")
                for lindex, label_index, label in zip(range(len(label_names)), labels_index, label_names):
                    for cmp_index in range(n_components):
                        samples1 = np.linspace(means[findex, lindex, cmp_index] - 3*stds[findex, lindex, cmp_index], means[findex, lindex, cmp_index] + 3*stds[findex, lindex, cmp_index], 100)
                        axs[row, col].plot(samples1, _norm(samples1, means[findex, lindex, cmp_index], stds[findex, lindex, cmp_index]), c=color[lindex])
                    axs[row, col].plot(X[y == label_index, findex], np.zeros((np.sum(y == label_index))), c=color[lindex], marker='x', markersize=5, linestyle = 'None', label=label)
                    #axs[row, col].legend(loc="upper right") 
        plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
        plt.savefig(path_dir + f'multinorm{idx}.png', bbox_inches='tight')  
def plot_one_data(x, y, means, stds): 
    """Plot one normal distribution graph
    Args:
        X (_array (N_samples, N_features)_): dataset
        y (_N_features_): labels
        means (array_N_features, 2, n_components_): mean for each features
        stds (array_N_features, 2, n_components_): std for each features
    """
    
    if means.shape[1] != 2:
        raise Exception(f"We only plot for two labels for now ")
    
    if means.shape != stds.shape:
        raise Exception(f"Number of means {means.shape} different from stds {stds.shape} ")
    
    fig, axs = plt.subplots(nrows=1, ncols=1) 
   
    findex = 0
    axs.set_title(f"Model for feature's {findex}th")
    for cmp_index in range(means.shape[2]):
        samples1 = np.linspace(means[findex, 0, cmp_index] - 3*stds[findex, 0, cmp_index], means[findex, 0, cmp_index] + 3*stds[findex, 0, cmp_index], 100)
        axs.plot(samples1, _norm(samples1, means[findex, 0, cmp_index], stds[findex, 0, cmp_index]), c='b')
    for cmp_index in range(means.shape[2]):
        samples2 = np.linspace(means[findex, 1, cmp_index] - 3*stds[findex, 1, cmp_index], means[findex, 1, cmp_index] + 3*stds[findex, 1, cmp_index], 100)
        axs.plot(samples2, _norm(samples2, means[findex, 1, cmp_index], stds[findex, 1, cmp_index]), c='r')
    if x is not None:
        if y == 1:
            axs.plot(x, 0, c='b', marker='o')
        else:
            axs.plot(x, 0, c='r', marker='o')
        
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
        raise Exception(f"We only plot for two labels for now ")
    
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
    
def plot_global_history(history, N_iter, N_states, path_dir):
    Si_fits = np.array(history["Si_fit"])
    Sg_fits = np.array(history["Sg_fit"])
    dtrs = np.array(history["DETR"])
    fars = np.array(history["FAR"])
    
    if Si_fits.shape != (N_iter, N_states):
        raise Exception(f"Expect Si_fits to be {(N_iter, N_states)}, but have {Si_fits.shape}")
    if Sg_fits.shape != (N_iter, ):
        raise Exception(f"Expect Sg_fits to be {(N_iter, )}, but have {Sg_fits.shape}")
    if dtrs.shape != (N_iter, N_states):
        raise Exception(f"Expect dtrs to be {(N_iter, N_states)}, but have {dtrs.shape}")
    if fars.shape != (N_iter, N_states):
        raise Exception(f"Expect fars to be {(N_iter, N_states)}, but have {fars.shape}")

    fig, axs = plt.subplots(nrows=2, ncols=1) 
    axs[0].set_title("Best-fit, Global fit")
    axs[0].plot(np.arange(N_iter), Sg_fits[:], linewidth=2, label="Global fitness")
    for Q_index in range(N_states):
        axs[0].plot(np.arange(N_iter), Si_fits[:, Q_index], linewidth=2, label=f"S{Q_index} fitness")
    axs[0].legend(loc="upper right")  
    
    axs[1].set_title("Detection rate, False alarm rate")
    for Q_index in range(N_states):
        axs[1].plot(np.arange(N_iter), dtrs[:, Q_index], linewidth=2, label=f"dtr {Q_index}")
        axs[1].plot(np.arange(N_iter), fars[:, Q_index], linewidth=2, label=f"far {Q_index}")
    axs[1].legend(loc="upper right") 
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    plt.legend()
    plt.savefig(path_dir + f'history.png', bbox_inches='tight')  
    
    
def roc_curve_plot(y_test, y_score, path_dir):
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_area = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_area,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(path_dir + 'roc_curve.png', bbox_inches='tight')
    return fpr, tpr

def plot_density_line(y_preds):
    """Plot density line for one labels

    Args:
        y_preds (_(N_samples,)_): {-1, 1}
    """
    N = y_preds.shape[0]
    X = np.arange(N)
    
    fig,ax = plt.subplots()
    plt.xlim([-5.0, N + 5])
    plt.ylim([-1.2, 1.2])
    ax.plot(X, y_preds, c='r', marker="D",  markersize=5, linestyle = 'None')
    ax.plot(X, np.ones(N, ))
    ax.plot(X, -1*np.ones(N, ))
    ax.set_title("Figure:")
    ax.legend(loc="lower right")
    plt.show()
    