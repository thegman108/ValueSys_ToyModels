import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import os

# Set path to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Loading in test datasets and model tested
subset_index = 1000
with open('sparse_random_activations_lora_orca_A1.pkl', 'rb') as f:
    sparse_activations_orca = pickle.load(f)[:subset_index]
    print("Sparse Activations Shape:", sparse_activations_orca.shape)
        
with open('dense_activations_lora_orca_A1.pkl', 'rb') as f:
    dense_activations_orca = pickle.load(f)[:subset_index]
    print("Dense Activations Shape:", dense_activations_orca.shape)

with open('orca_lr_model_updated_llama7b.pkl', 'rb') as f:
    model = pickle.load(f)
    print("Model Loaded")

# Test model accuracy on sparse activations
y_true_sparse = [0] * subset_index
sparse_predictions = model.predict(sparse_activations_orca)
sparse_accuracy = accuracy_score(sparse_predictions, y_true_sparse)
print("Sparse Activations Accuracy:", sparse_accuracy)

# Test model accuracy on dense activations
y_true_dense = [1] * subset_index
dense_predictions = model.predict(dense_activations_orca)
dense_accuracy = accuracy_score(dense_predictions, y_true_dense)
print("Dense Activations Accuracy:", dense_accuracy)