from gmms import OnlineGMM
import numpy as np
import matplotlib.pyplot as plt


N_samples = 200
N_test = 30
#TRAINING SET
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_samples,))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_samples,))
X = np.concatenate((X0, X1))
Y = np.concatenate((np.ones((N_samples, ), dtype=np.int32),
                    -1*np.ones((N_samples, ), dtype=np.int32)))
train_normal_mean = np.mean(X[Y==1])
train_normal_std = np.sqrt(np.sum((X[Y==1] - train_normal_mean)**2 / np.sum(Y==1)))

train_attack_mean = np.mean(X[Y==-1])
train_attack_std = np.sqrt(np.sum((X[Y==-1] - train_attack_mean)**2 / np.sum(Y==-1)))

#TESTING SET
X0 = np.random.normal(loc=2.0, scale=0.3, size=(N_test,))
X1 = np.random.normal(loc=4.0, scale=0.3, size=(N_test,))
X_test = np.concatenate((X0, X1), axis=0)
Y_test = np.concatenate((np.ones((N_test, ), dtype=np.int32), 
                        -1*np.ones((N_test, ), dtype=np.int32)))

# plt.scatter(np.arange(50), X)
# plt.show()
model = OnlineGMM(std=1.0, n_components=4, T=0.5)
model.build(n_labels=2)

    # print(">> BEFORE TRAIN")
    # print(f"Means: {model.means}")
    # print(f"Stds: {model.stds}")
    # print(f"Weight: {model.weight}")
    # print(f"N: {model.N}")
    # print(f"Predict: {model.predict(X[2])}")

for x, y in zip(X, Y):
    print(f"x: {x}, y: {y}")
    model.fit(x, y)
print(">> AFTER TRAIN")
print(f">>train mean: {train_normal_mean}")
print(f">>train std: {train_normal_std}")
print(f">>test mean: {train_normal_mean}")
print(f">>test std: {train_normal_std}")

# print(f"Means: {model.means}")
# print(f"Stds: {model.stds}")
# print(f"Weight: {model.weight}")
# print(f"N: {model.N}")
# print(f"Predict: {model.predict(X[2])}")

predicts = model.predict(X_test)

print(f"labels: {Y_test}")
print(f"predic: {predicts}")

def detection_rate(labels, predicts):
    """Calculate detection rate (TPR)

    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"TP: {np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32)}")
    # print(f"TP + FN: {np.sum(labels == -1)}")
    return np.sum((labels - predicts)[labels == -1] == 0, dtype=np.float32) / np.sum(labels == -1)

def false_alarm_rate(labels, predicts):
    """Calculate false alarm rate
    Args:
        labels (_array-like (N_test,)_): _label of test dataset_
        predicts (_array-like (N_test_): _label predicted by model_
    """
    # print(f"FP: {np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32)}")
    # print(f"FP + TN: {np.sum(labels == 1)}")
    return np.sum((labels - predicts)[labels == 1] != 0, dtype=np.float32) / np.sum(labels == 1)

dtr = detection_rate(Y_test, predicts)
flr = false_alarm_rate(Y_test, predicts)

print(f">> detection rate: {dtr}")
print(f">> false alarm rate: {flr}")

