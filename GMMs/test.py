from gmms import OnlineGMM
import numpy as np
import matplotlib.pyplot as plt

X = np.random.normal(0, 1, 50)
# plt.scatter(np.arange(50), X)
# plt.show()
model = OnlineGMM(n_components=4, T=0.5)
model.build(n_labels=4)

print(">> BEFORE TRAIN")
print(f"Means: {model.means}")
print(f"Stds: {model.stds}")
print(f"Weight: {model.weight}")
print(f"N: {model.N}")
print(f"Predict: {model.predict(X[2])}")

for i in range(50):
    model.fit(X[i])
    
print(">> AFTER TRAIN")
print(f"Means: {model.means}")
print(f"Stds: {model.stds}")
print(f"Weight: {model.weight}")
print(f"N: {model.N}")
print(f"Predict: {model.predict(X[2])}")