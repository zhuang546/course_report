import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gdown
from sklearn.metrics import roc_curve, roc_auc_score

# Download data files
if not os.path.isfile('u.data'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    gdown.download(url, 'u.data', quiet=False)

df = pd.read_table('u.data', names=('user_id', 'item_id', 'rating', 'timestamp'))
X_df = df.pivot_table(index='user_id', columns='item_id', values='rating')

if not os.path.isfile('u1.base'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u1.base'
    gdown.download(url, 'u1.base', quiet=False)
if not os.path.isfile('u1.test'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/u1.test'
    gdown.download(url, 'u1.test', quiet=False)

train_df = pd.read_table('u1.base', names=('user_id', 'item_id', 'rating', 'timestamp'))
X_train_df = train_df.pivot_table(index='user_id', columns='item_id', values='rating')

for c in X_df.columns:
    if c not in X_train_df.columns:
        X_train_df[c] = np.nan
X_train_df = X_train_df.sort_index(axis=1)

X_train = X_train_df.values
print('Training rating matrix X_train:')
print(X_train)

R_train_df = X_train_df.T.corr()
R_train = R_train_df.values
print('User-user correlation matrix R_train:')
print(R_train)

test_df = pd.read_table('u1.test', names=('user_id', 'item_id', 'rating', 'timestamp'))
print('Test data:')
print(test_df)

D_train = X_train - np.nanmean(X_train, axis=1)[:, np.newaxis]
user_mean = np.nanmean(X_train, axis=1, keepdims=True)

mask = ~np.isnan(D_train)               # users who actually rated each item
D0 = np.nan_to_num(D_train, nan=0.0)    # rating deviations (NaN → 0)
R0 = np.nan_to_num(R_train, nan=0.0)    # correlations (NaN → 0)

numer = R0 @ D0                         # Σ_j r_ij · (r_jk − r̄_j)
denom = np.abs(R0) @ mask.astype(float) # Σ_j |r_ij| over valid ratings

with np.errstate(divide='ignore', invalid='ignore'):
    Y_pred = user_mean + numer / denom
Y_pred = np.where(np.isfinite(Y_pred), Y_pred, user_mean)

u_idx = test_df['user_id'].to_numpy() - 1
i_idx = test_df['item_id'].to_numpy() - 1
true_rate = test_df['rating'].to_numpy()
y_pred = Y_pred[u_idx, i_idx]

result = np.vstack([true_rate, y_pred]).T
print('True ratings and predicted ratings:')
print(result)

fpr, tpr, thresholds = roc_curve(true_rate >= 2, y_pred)
plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid()
plt.show()

auc_val = roc_auc_score(true_rate >= 2, y_pred)
print('ROC-AUC score:', auc_val)

