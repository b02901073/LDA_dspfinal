from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import csv
import sys
import lda
import matplotlib.pyplot as plt
from time import time

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

n_top_words = 3

data = []
check_index = []

with open('title_StackOverflow.txt') as f:
    data = f.readlines()
data = np.array(data)

with open('check_index.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        check_index.append(row)
check_index = np.array(check_index)
check_index = np.delete(check_index,0,0)

ans = []
with open('label_StackOverflow.txt') as f:
    reader = csv.reader(f)
    for row in reader:
        ans.append(row)
ans = np.array(ans)

### METHOD 2
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=None,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(data)
print("done in %0.3fs." % (time() - t0))

model = lda.LDA(n_topics=20, n_iter=2000, random_state=3)
model.fit(tf)

n_top_words = 3
topic_word = model.topic_word_  # model.components_ also works
for i, topic_dist in enumerate(topic_word):
    topic_words = np.asarray(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


doc_topic = model.doc_topic_
print(doc_topic.shape)
print(ans.shape)

count = 0
with open('ans.csv','w') as f:
    w = csv.writer(f)
    w.writerows([['ID','Ans']])
    for i in range(check_index.shape[0]):
        a = int(check_index[i][1])
        b = int(check_index[i][2])
        if doc_topic[a].argmax() == doc_topic[b].argmax():
            w.writerows([[i,1]])
            if ans[a] == ans[b]:
                count += 1
        else:
            w.writerows([[i,0]])
            if ans[a] != ans[b]:
                count += 1

print(count)
print('acc:',float(count)/float(check_index.shape[0]))

plt.plot(model.loglikelihoods_[5:])
plt.savefig('ml_hw4.png')

'''
### METHOD 1
vectorizer = TfidfVectorizer(max_df=0.5, max_features=None, min_df=2, stop_words='english')
X = vectorizer.fit_transform(data)



svd = TruncatedSVD(n_components=20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)

#K-means
km = KMeans(n_clusters=20, init='k-means++', n_init=20, verbose=False).fit(X)
#print('end')
print(X.shape)
L = km.labels_
print(L.shape)

count = 0
for i in range(check_index.shape[0]):
    a = int(check_index[i][1])
    b = int(check_index[i][2])
    if L[a] == L[b] and ans[a] == ans[b]:
        count += 1
    elif L[a] != L[b] and ans[a] != ans[b]:
        count += 1
print(count)
print('acc:',float(count)/float(check_index.shape[0]))
'''