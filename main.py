# This is refered by anokas's Data Analysis & XGBoost Starter (0.35460 LB)
# https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb

import gensim
import nltk
import numpy as np
import pandas as pd
import sys

from collections import Counter
from custom_library.utils import get_current_millis, get_elapsed_seconds
from feature_extracter import FeatureExtracter
from oversampler import OverSampler
from time_logger import TimeLogger
from sklearn import cross_validation
from xgb_manager import XgbManager

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  # Input file path
  parser.add_option('--testFile', dest='test_file')
  parser.add_option('--trainFile', dest='train_file')
  parser.add_option('--refinedTrainFile', dest='refined_train_file')
  parser.add_option('--refinedTestFile', dest='refined_test_file')
  parser.add_option('--wordVectorFile', dest='word_vector_file')
  # Output file path
  parser.add_option('--submissionFile', dest='submission_file')

  options, otherjunk = parser.parse_args(argv)
  return options

arguments = parse_commands(sys.argv[1:])
time_logger = TimeLogger()

# Step00. Gets word vector.
time_logger.start()
word_vector_model = gensim.models.KeyedVectors.load(arguments.word_vector_file)
time_logger.log_with_elapse('Step00, Load word vector model time:')

# Step0. Gets train data and test data
time_logger.start()
train_data = pd.read_csv(arguments.train_file)
test_data = pd.read_csv(arguments.test_file)
refined_train_data = pd.read_csv(arguments.refined_train_file)
refined_test_data = pd.read_csv(arguments.refined_test_file)
train_questions = pd.Series(train_data['question1'].tolist() + train_data['question2'].tolist()).astype(str)
test_questions = pd.Series(test_data['question1'].tolist() + test_data['question2'].tolist()).astype(str)
time_logger.log_with_elapse('Step0, Get train, test data time:')

fe = FeatureExtracter(train_questions, word_vector_model)

# Step1. Reblancing data
# First we create our training and testing data
time_logger.start()
X_train = fe.get_features(train_data, refined_train_data)
X_test = fe.get_features(test_data, refined_test_data)
y_train = train_data['is_duplicate'].values
pos_train = X_train[y_train == 1]
neg_train = X_train[y_train == 0]
time_logger.log_with_elapse('Step1, Rebalancing data time:')

# Step2. Over sampling
# Now we oversample the negative class
# There is likely a much more elegant way to do this...
time_logger.start()
oversampler = OverSampler(pos_train, neg_train)
X_train, y_train = oversampler.get_over_sample()
time_logger.log_with_elapse('Step2, Over sampling time:')

# Step3. Split train data to create validation data
# Finally, we split some of the data off for validation
time_logger.start()
X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(X_train, y_train, test_size=0.2, random_state=4242)
time_logger.log_with_elapse('Step3, Split validation data from train data time:')

# Step4. Run XGBoost algorithm.
time_logger.start()
xgb = XgbManager(X_train, X_valid, X_test, y_train, y_valid)
# Set our parameters for xgboost
params = {
  'objective': 'binary:logistic',
  'eval_metric': 'logloss',
  'eta': 0.02,
  'max_depth': 4,
}
# Set our parametic options for xgboost
options = {
  # Activates early stopping.
  'early_stopping_rounds': 50,
  # The evaluation metric on the validation set is printed at every given verbose_eval boosting stage.
  'verbose_eval': 10,
}
# Number of boosting iterations.
num_boost_round = 400
xgb.train(params, options, num_boost_round)
y_test = xgb.predict()
time_logger.log_with_elapse('Step4, Run XGBoost time:')

# Step5. Make submission file.
time_logger.start()
sub = pd.DataFrame()
sub['test_id'] = test_data['test_id']
sub['is_duplicate'] = y_test
sub.to_csv(arguments.submission_file, index=False)
time_logger.log_with_elapse('Step5, Make submission time:')
print('Done.')