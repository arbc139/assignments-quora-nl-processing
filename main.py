
import gensim
import nltk
import numpy as np
import string
import sys

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
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
    
    print('not stopwords:', subtree[0])
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
      if not word1 in model.vocab or not word2 in model.vocab:
        continue
      similarity = model.similarity(word1, word2)
      if similarity > max_similarity:
        max_similarity = similarity
  features['max_similarity'] = max_similarity

  # Third Feature. Count of similarity words
  alpha = 0.6
  count = 0
  for word1 in sentence1:
    for word2 in sentence2:
      if not word1 in model.vocab or not word2 in model.vocab:
        continue
      similarity = model.similarity(word1, word2)
      if similarity >= alpha:
        count += 1
  features['similarity_count'] = count / (len(sentence1) * len(sentence2))

  print(features)
  print(features.values())
  return features
  

model = gensim.models.KeyedVectors.load_word2vec_format(options.model_path, binary=True)
STOPWORDS = nltk.corpus.stopwords.words('english')

sentence1 = 'Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?'
sentence1_words = filter_NNP_from_chunk_tree(
  STOPWORDS,
  nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence1)))
)
sentence2 = 'How does the Surface Pro himself 4 compare with iPad Pro?'
sentence2_words = filter_NNP_from_chunk_tree(
  STOPWORDS,
  nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence2)))
)

get_features(model, sentence1_words, sentence2_words)