import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from torch.quantization import quantize_dynamic
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = mnist_trainset.data.numpy().reshape(len(mnist_trainset), -1)
y_train = mnist_trainset.targets.numpy()
X_test = mnist_testset.data.numpy().reshape(len(mnist_testset), -1)
y_test = mnist_testset.targets.numpy()

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Model Accuracy:", accuracy)

import joblib
import os

joblib.dump(logistic_model, "logistic_model.pkl")
model_size = os.path.getsize("logistic_model.pkl") / 1024
print("Logistic Regression Model Size:", model_size, "KB")

import time

start_time = time.time()
logistic_model.predict(X_test[:100])
inference_time = (time.time() - start_time) / 100
print("Logistic Regression Model Inference Time (per sample):", inference_time, "seconds")

def quantize_model(model, scale_factor=2**7):
    model.coef_ = (model.coef_ * scale_factor).astype(np.int8)
    model.intercept_ = (model.intercept_ * scale_factor).astype(np.int8)
    return model

def quantized_inference(model, X):
    X_quantized = (X / 2**7).astype(np.float32)
    return model.predict(X_quantized)

quantized_model = quantize_model(logistic_model)
y_quantized_pred = quantized_inference(quantized_model, X_test)
quantized_accuracy = accuracy_score(y_test, y_quantized_pred)
print("Quantized Model Accuracy:", quantized_accuracy)

joblib.dump(quantized_model, "quantized_model.pkl")
quantized_model_size = os.path.getsize("quantized_model.pkl") / 1024
print("Quantized Model Size:", quantized_model_size, "KB")

start_time = time.time()
quantized_inference(quantized_model, X_test[:100])
quantized_inference_time = (time.time() - start_time) / 100
print("Quantized Model Inference Time (per sample):", quantized_inference_time, "seconds")

print("\nComparison Results:")
print("Original Model Size:", model_size, "KB")
print("Quantized Model Size:", quantized_model_size, "KB")
print("Original Inference Time:", inference_time, "seconds per sample")
print("Quantized Inference Time:", quantized_inference_time, "seconds per sample")
print("Original Model Accuracy:", accuracy)
print("Quantized Model Accuracy:", quantized_accuracy)