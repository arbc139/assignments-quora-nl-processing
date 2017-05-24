# This is refered by anokas's Data Analysis & XGBoost Starter (0.35460 LB)
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb

import gc
import nltk
import numpy as np
import os
import pandas as pd
import sys

from collections import Counter
from sklearn import cross_validation

STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('--testFile', dest='test_file')
  parser.add_option('--trainFile', dest='train_file')
  parser.add_option('--submissionFile', dest='submission_file')

  options, otherjunk = parser.parse_args(argv)
  return options

options = parse_commands(sys.argv[1:])

# Step0. Gets train data and test data
train_data = pd.read_csv(options.train_file)
test_data = pd.read_csv(options.test_file)

train_questions = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
test_questions = pd.Series(test_data['question1'].tolist() + test_data['question2'].tolist()).astype(str)

# Step1. Calculates word match feature.
def word_match_share(row):
  question1_words = {}
  question2_words = {}
  for word in str(row['question1']).lower().split():
    if word not in STOPWORDS:
      question1_words[word] = 1
  for word in str(row['question2']).lower().split():
    if word not in STOPWORDS:
      question2_words[word] = 1
  if len(question1_words) == 0 or len(question2_words) == 0:
    # The computer-generated chaff includes a few questions that are nothing but stopwords
    return 0
  shared_words_in_q1 = [
    word for word in question1_words.keys() if word in question2_words
  ]
  shared_words_in_q2 = [
    word for word in question2_words.keys() if word in question1_words
  ]
  return (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(question1_words) + len(question2_words))

# Step2. Calculates train word weights feature.
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
  if count < min_count:
    return 0
  return 1 / (count + eps)

eps = 5000
words = (' '.join(train_questions)).lower().split()
counts = Counter(words)
weights = { word: get_weight(count) for word, count in counts.items() }

# Step3. Calculates tf-idf weights feature.
def tfidf_word_match_share(row):
  question1_words = {}
  question2_words = {}
  for word in str(row['question1']).lower().split():
    if word not in STOPWORDS:
      question1_words[word] = 1
  for word in str(row['question2']).lower().split():
    if word not in STOPWORDS:
      question2_words[word] = 1
  if len(question1_words) == 0 or len(question2_words) == 0:
    # The computer-generated chaff includes a few questions that are nothing but stopwords
    return 0
  
  shared_weights = [
    weights.get(word, 0) for word in question1_words.keys() if word in question2_words # Share words with q2 in q1
  ] + [
    weights.get(word, 0) for word in question2_words.keys() if word in question1_words # Share words with q1 in q2
  ]
  total_weights = [
    weights.get(word, 0) for word in question1_words
  ] + [
    weights.get(word, 0) for word in question2_words
  ]
  
  return np.sum(shared_weights) / np.sum(total_weights)

# Step4. Reblancing data
# First we create our training and testing data
X_train = pd.DataFrame()
X_test = pd.DataFrame()
X_train['word_match'] = train_data.apply(word_match_share, axis=1, raw=True)
X_train['tfidf_word_match'] = train_data.apply(tfidf_word_match_share, axis=1, raw=True)
X_test['word_match'] = test_data.apply(word_match_share, axis=1, raw=True)
X_test['tfidf_word_match'] = test_data.apply(tfidf_word_match_share, axis=1, raw=True)

y_train = train_data['is_duplicate'].values

pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]

# Step5. Over sampling
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

X_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train

# Step6. Split train data to create validation data
# Finally, we split some of the data off for validation
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train, y_train, test_size=0.2, random_state=4242)

# Step7. Run XGBoost algorithm.
import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)

d_test = xgb.DMatrix(X_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = p_test
sub.to_csv(options.submission_file, index=False)