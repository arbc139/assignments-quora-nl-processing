
import gensim
import nltk
import numpy as np
import string
import sys
import time

from csv_manager import CsvWriter, CsvReader
from sklearn import svm

get_current_millis = lambda: int(round(time.time() * 1000))
def get_elapsed_seconds(current_time, elapsed_millis):
  return (current_time - elapsed_millis) / 1000.0

elapsed_millis = get_current_millis()
STOPWORDS = nltk.corpus.stopwords.words('english')
print('Get stop word time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('--testFile', dest='test_file')
  parser.add_option('--trainFile', dest='train_file')
  parser.add_option('-o', '--submissionFile', dest='submission_file')
  parser.add_option('-m', '--modelPath', dest='model_path')

  options, otherjunk = parser.parse_args(argv)
  return options

options = parse_commands(sys.argv[1:])

def get_tokens(sentence):
  # Make lower, remove punctuation.
  processed_sentence = sentence.lower().translate(
    dict.fromkeys(map(ord, string.punctuation), None)
  )
  return nltk.word_tokenize(processed_sentence)

def filter_stop_words(tokens):
  return [word for word in tokens if not word in nltk.corpus.stopwords.words('english')]

def filter_NNP_from_chunk_tree(stopwords, tree):
  result = []
  for subtree in tree:
    if type(subtree) != nltk.tree.Tree and \
      (subtree[0].lower() in stopwords or subtree[1] == '.'):
      continue
    
    if type(subtree) == nltk.tree.Tree:
      subtree_result = filter_NNP_from_chunk_tree(stopwords, subtree)
      result.append('_'.join(subtree_result))
      continue
    
    result.append(subtree[0])
  return result

def get_word_vector(model, word):
  try:
    return model[word]
  except KeyError:
    return np.zeros(300)
  
def normalize_vector(vector):
  magnitude = np.linalg.norm(vector)
  if magnitude == 0:
    return vector
  return vector / magnitude

def cosine_distance(vector1, vector2):
  magnitude1 = np.linalg.norm(vector1)
  magnitude2 = np.linalg.norm(vector2)
  if magnitude1 * magnitude2 == 0:
    return 0
  return np.dot(vector1, vector2) / (magnitude1 * magnitude2)

def filter_words_not_in_model(model, words):
  return list(filter(lambda word: word in model.vocab, words))

def get_features(model, sentence1, sentence2):
  features = dict()
  
  # First feature. Total sentence similarity.
  sentence1_vector = np.zeros(300)
  for word in sentence1:
    sentence1_vector = np.add(sentence1_vector, get_word_vector(model, word))
  normalized_sentence1_vector = normalize_vector(sentence1_vector)
  
  sentence2_vector = np.zeros(300)
  for word in sentence2:
    sentence2_vector = np.add(sentence2_vector, get_word_vector(model, word))
  normalized_sentence2_vector = normalize_vector(sentence2_vector)

  features['sentence_similarity'] = cosine_distance(sentence1_vector, sentence2_vector)
  
  # Second Feature. Max similarity.
  max_similarity = 0
  for word1 in sentence1:
    for word2 in sentence2:
      similarity = model.similarity(word1, word2)
      if similarity > max_similarity:
        max_similarity = similarity
  features['max_similarity'] = max_similarity

  # Third Feature. Count of similarity words
  alpha = 0.6
  count = 0
  for word1 in sentence1:
    for word2 in sentence2:
      similarity = model.similarity(word1, word2)
      if similarity >= alpha:
        count += 1
  if len(sentence1) == 0 or len(sentence2) == 0:
    features['similarity_count'] = 0
  else:
    features['similarity_count'] = count / (len(sentence1) * len(sentence2))

  return features

def make_sentences_to_features(model, sentence1, sentence2):
  # sentence1 = 'Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?'
  sentence1_words = filter_words_not_in_model(model, filter_NNP_from_chunk_tree(
    STOPWORDS,
    nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence1)))
  ))
  # sentence2 = 'How does the Surface Pro himself 4 compare with iPad Pro?'
  sentence2_words = filter_words_not_in_model(model, filter_NNP_from_chunk_tree(
    STOPWORDS,
    nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence2)))
  ))

  return get_features(model, sentence1_words, sentence2_words)

elapsed_millis = get_current_millis()
model = gensim.models.KeyedVectors.load_word2vec_format(options.model_path, binary=True)
print('Making word2vec model time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
train_data = None
with open(options.train_file, 'r') as train_file:
  csv_reader = CsvReader(train_file)
  train_data = csv_reader.get_dict_list_data()
print('Get train data time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
test_data = None
with open(options.test_file, 'r') as test_file:
  csv_reader = CsvReader(test_file)
  test_data = csv_reader.get_dict_list_data()
print('Get test data time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
train_features = []
train_results = []
for train in train_data:
  train_features.append(make_sentences_to_features(model, train['question1'], train['question2']))
  train_results.append(int(train['is_duplicate']))
print('Convert train data to features time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
svc = svm.SVC(kernel='linear')
svc.fit(train_features, train_results)
print('Train svm time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
test_features = []
for test in test_data:
  test_features.append(make_sentences_to_features(model, test['question1'], test['question2']))
print('Convert test data to features time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
test_results = svc.predict(test_features)
print('Predict test data time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
with open(options.submission_file, 'w+') as submission_file:
  csv_writer = CsvWriter(submission_file, ['test_id', 'is_duplicate'])
  csv_writer.write_header()
  for result in test_results:
    csv_writer.write_row({
      'test_id': test['test_id'],
      'is_duplicate': result,
    })
print('Write submission file time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))
