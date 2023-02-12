import gensim.downloader as api

## Load 100-D word vecs
D = 100
word_vectors = api.load("glove-wiki-gigaword-100")

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

## Generate word-pair vectors as input data
# p_min: minimum similarity between the words of a pair (0.8-0.95)
# co_max: maximum number of similar words if there is enough (8)
# N_train: size of training set (1900)
# N_test: size of test set (100)
p_min, co_max = .9, 8
N_train, N_test = 1000, 100
x_train, y_train = [], []
x_test, y_test = [], []
i = 0
for w in word_vectors.index2entity:
  if len(w)<3:
    continue
  cowords = word_vectors.similar_by_word(w)
  cowords = cowords[:min(len(cowords), co_max)]
  for sim in cowords:
    v, p = sim
    if p<=p_min:
      break
    if len(v)<3:
      continue
    i+=1
    if i<N_train:
      x_train.append(word_vectors.get_vector(w))
      y_train.append(word_vectors.get_vector(v))
    elif i<N_train+N_test:
      x_test.append(word_vectors.get_vector(w))
      y_test.append(word_vectors.get_vector(v))
    else:
      break
  if i==N_train+N_test:
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    break

# Define the model
model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(D,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(D, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Fit the model on the training data
model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=0)

## Measure the R2-score of the model
# N: size of the test set
# T: time of measurements
from sklearn.metrics import r2_score
import random
N = 100
T = 20
y_true, y_pred = [], []
R2s = []
for _ in range(T):
  x_test_shuffled = [ random.choice(x_test) for _ in range(N) ]
  y_true = [ word_vectors.get_vector(word_vectors.similar_by_vector(v)[0][0]) for v in x_test_shuffled ]
  y_pred = [ model.predict(np.array([v]), verbose=0)[0] for v in x_test_shuffled ]
  R2s.append(r2_score(y_true, y_pred))
print('R2-scores:\nmin:', min(R2s)*100,'%, max:',max(R2s)*100,'%, avg:',sum(R2s)/T*100,'%, std:',np.std(np.array(R2s))*100,'%')
