print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from gensim.models import word2vec
import pickle


def get_words_for_label(model, words, labels, label):
  wordsl = []
  labelsl = np.argwhere(labels==label)
  for lbsli in labelsl:
    wordsl.append(words[lbsli[0]][0].encode('ascii', 'ignore'))

  return wordsl

def get_label_for_word(model, wordsonly, labels, word):
  return labels[wordsonly.index(word)]


veclen = 256

model_file = '___path to the trained word2vec binary file___'

model = word2vec.Word2Vec.load_word2vec_format(model_file, binary=True)


vocab_size = len(model.vocab)

wordvecs_vecs = np.empty([vocab_size, veclen]) * np.nan

wordvecs_words = model.vocab.items()

wordvecs_wordsonly = model.vocab.keys()


for i in range(len(wordvecs_words)):
  wordvecs_vecs[i] = model.syn0norm[model.vocab[wordvecs_words[i][0]].index]


n_clusters = 1000
n_init = 10
init_size = 3*n_clusters
batch_size = 100
max_no_improvement = 10
n_jobs = 10

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init, n_jobs=n_jobs)
kmeans.fit(wordvecs_vecs)

mbkmeans = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, 
  batch_size=batch_size, n_init=n_init, max_no_improvement=max_no_improvement, 
  init_size=init_size, verbose=0)
mbkmeans.fit(wordvecs_vecs)


wordvecs_labels = mbkmeans.labels_

print get_words_for_label(model, wordvecs_words, wordvecs_labels, 0)
print get_label_for_word(model, wordvecs_wordsonly, wordvecs_labels, 'liver')
