from GMMs.gmms import OnlineGMM
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import detection_rate, false_alarm_rate, load_fake_data
from visualize import plot_one_norm_data
from tqdm import tqdm


model = OnlineGMM(std=1.0, n_components=4, T=0.5)
model.build(n_labels=2)

N_train = 100
N_test = 100
X_train, X_test, Y_train, Y_test = load_fake_data(N_train, N_test, 1)

#Visualize:
print(">> Visualize before training ...")
means = model.means[None, :, :]
stds = model.stds[None, :, :]
plot_one_norm_data(X_train, Y_train, means, stds)

print(">> Training ...")
for x, y in tqdm(zip(X_train, Y_train), total=2*N_train):
    model.fit(x, y)
    
#print(">> AFTER TRAIN")
#Visualize:
print(">> Visualize after training ...")
means = model.means[None, :, :]
stds = model.stds[None, :, :]
plot_one_norm_data(X_train, Y_train, means, stds)

print("=====================================")

