from gmms import OnlineGMM
import numpy as np

X = np.random.normal(0, 1, 50)
model = OnlineGMM(n_components=16, T=0.5)
model.build(n_labels=3)
for i in range(50):
    model.fit(X[i])
print(model.predict_prob(X[2]))
print(model.weight)
print(model.stds)
print(model.means)
print(model.N)
print(model.A)

