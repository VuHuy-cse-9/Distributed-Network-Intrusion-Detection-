from gmms import OnlineGMM
import numpy as np
import matplotlib.pyplot as plt


N_samples = 50
X = np.arange(50)
X = np.random.normal(loc=0, scale=2, size=50)

# plt.scatter(np.arange(50), X)
# plt.show()
model = OnlineGMM(std=1.0, n_components=4, T=1)
model.build(n_labels=4)

    # print(">> BEFORE TRAIN")
    # print(f"Means: {model.means}")
    # print(f"Stds: {model.stds}")
    # print(f"Weight: {model.weight}")
    # print(f"N: {model.N}")
    # print(f"Predict: {model.predict(X[2])}")

for i in range(N_samples):
    model.fit(X[i])
    
print(">> AFTER TRAIN")
print(f"Means: {model.means}")
print(f"Stds: {model.stds}")
print(f"Weight: {model.weight}")
print(f"N: {model.N}")
print(f"Predict: {model.predict(X[2])}")