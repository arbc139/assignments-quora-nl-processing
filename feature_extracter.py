
import nltk
import numpy as np
import pandas as pd

from collections import Counter
from custom_library.utils import get_word_vector, normalize_vector, cosine_distance

stopwords = set(nltk.corpus.stopwords.words('english'))

class FeatureExtracter():
  def __init__(self, train_questions, word_vector_model):
    words = (' '.join(train_questions)).lower().split()
    counts = Counter(words)
    
    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(count, eps=10000, min_count=2):
      if count < min_count:
        return 0
      return 1 / (count + eps)

    # Step2. Calculates train word weights feature.
    self.word_weights = { word: get_weight(count) for word, count in counts.items() }
    self.word_vector_model = word_vector_model
  
  def get_features(self, data, refined_data):
    X = pd.DataFrame()

    # Feature1. Shard word match feature.
    def shared_word_match(row):
      question1_words = {}
      question2_words = {}
      for word in str(row['question1']).lower().split():
        if word not in stopwords:
          question1_words[word] = 1
      for word in str(row['question2']).lower().split():
        if word not in stopwords:
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
    X['word_match'] = data.apply(shared_word_match, axis=1, raw=True)

    # Feature1. Shard word match feature without stopwords.
    def shared_word_match_stop(row):
      question1_words = {}
      question2_words = {}
      for word in str(row['question1']).lower().split():
        question1_words[word] = 1
      for word in str(row['question2']).lower().split():
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
    X['word_match_stop'] = data.apply(shared_word_match, axis=1, raw=True)

    

    # Feature2. tf-idf word_weight feature.
    def tfidf_word_match_weight(row):
      question1_words = {}
      question2_words = {}
      for word in str(row['question1']).lower().split():
        question1_words[word] = 1
      for word in str(row['question2']).lower().split():
        question2_words[word] = 1
      if len(question1_words) == 0 or len(question2_words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
      shared_weights = [
        self.word_weights.get(word, 0) for word in question1_words.keys() if word in question2_words # Share words with q2 in q1
      ] + [
        self.word_weights.get(word, 0) for word in question2_words.keys() if word in question1_words # Share words with q1 in q2
      ]
      total_weights = [
        self.word_weights.get(word, 0) for word in question1_words
      ] + [
        self.word_weights.get(word, 0) for word in question2_words
      ]
      return np.sum(shared_weights) / np.sum(total_weights)
    X['tfidf_word_match'] = data.apply(tfidf_word_match_weight, axis=1, raw=True)

    # Feature3. tf-idf word_weight feature without stopwords.
    def tfidf_word_match_weight_stop(row):
      question1_words = {}
      question2_words = {}
      for word in str(row['question1']).lower().split():
        if word not in stopwords:
          question1_words[word] = 1
      for word in str(row['question2']).lower().split():
        if word not in stopwords:
          question2_words[word] = 1
      if len(question1_words) == 0 or len(question2_words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
      shared_weights = [
        self.word_weights.get(word, 0) for word in question1_words.keys() if word in question2_words # Share words with q2 in q1
      ] + [
        self.word_weights.get(word, 0) for word in question2_words.keys() if word in question1_words # Share words with q1 in q2
      ]
      total_weights = [
        self.word_weights.get(word, 0) for word in question1_words
      ] + [
        self.word_weights.get(word, 0) for word in question2_words
      ]
      return np.sum(shared_weights) / np.sum(total_weights)
    X['tfidf_word_match_stop'] = data.apply(tfidf_word_match_weight_stop, axis=1, raw=True)

    # Feature4. word vector similarity
    def word_vector_similarity(row):
      question1_vector = np.zeros(300)
      for word in str(row['question1']).lower().split():
        question1_vector = np.add(question1_vector, get_word_vector(self.word_vector_model, word))
      normalized_question1_vector = normalize_vector(question1_vector)
      question2_vector = np.zeros(300)
      for word in str(row['question2']).lower().split():
        question2_vector = np.add(question2_vector, get_word_vector(self.word_vector_model, word))
      normalized_question2_vector = normalize_vector(question2_vector)
      return cosine_distance(question1_vector, question2_vector)
    X['word_vector_similarity'] = refined_data.apply(word_vector_similarity, axis=1, raw=True)

    # Feature etc..
    def jaccard(row):
      wic = set(row['question1']).intersection(set(row['question2']))
      uw = set(row['question1']).union(row['question2'])
      if len(uw) == 0:
          uw = [1]
      return (len(wic) / len(uw))
    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)

    def common_words(row):
      return len(set(row['question1']).intersection(set(row['question2'])))
    X['common_words'] = data.apply(common_words, axis=1, raw=True)

    def total_unique_words(row):
      return len(set(row['question1']).union(row['question2']))
    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)

    def total_unique_words_stop(row, stops=stopwords):
      return len([x for x in set(row['question1']).union(row['question2']) if x not in stops])
    X['total_unique_words_stop'] = data.apply(total_unique_words_stop, axis=1, raw=True)

    def wc_diff(row):
      return abs(len(row['question1']) - len(row['question2']))
    X['wc_diff'] = data.apply(wc_diff, axis=1, raw=True)

    def wc_ratio(row):
      l1 = len(row['question1'])*1.0 
      l2 = len(row['question2'])
      if l2 == 0:
        return np.nan
      if l1 / l2:
        return l2 / l1
      else:
        return l1 / l2
    X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)

    def wc_diff_unique(row):
      return abs(len(set(row['question1'])) - len(set(row['question2'])))
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)

    def wc_ratio_unique(row):
      l1 = len(set(row['question1'])) * 1.0
      l2 = len(set(row['question2']))
      if l2 == 0:
        return np.nan
      if l1 / l2:
        return l2 / l1
      else:
        return l1 / l2
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)

    def wc_diff_unique_stop(row, stops=stopwords):
      return abs(len([x for x in set(row['question1']) if x not in stops]) - len([x for x in set(row['question2']) if x not in stops]))
    X['wc_diff_unique_stop'] = data.apply(wc_diff_unique_stop, axis=1, raw=True)

    def wc_ratio_unique_stop(row, stops=stopwords):
      l1 = len([x for x in set(row['question1']) if x not in stops])*1.0 
      l2 = len([x for x in set(row['question2']) if x not in stops])
      if l2 == 0:
        return np.nan
      if l1 / l2:
        return l2 / l1
      else:
        return l1 / l2
    X['wc_ratio_unique_stop'] = data.apply(wc_ratio_unique_stop, axis=1, raw=True)

    def same_start_word(row):
      if not row['question1'] or not row['question2']:
        return np.nan
      return int(row['question1'][0] == row['question2'][0])
    X['same_start_word'] = data.apply(same_start_word, axis=1, raw=True)

    def char_diff(row):
      return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))
    X['char_diff'] = data.apply(char_diff, axis=1, raw=True)
    
    def char_diff_unique_stop(row, stops=stopwords):
      return abs(len(''.join([x for x in set(row['question1']) if x not in stops])) - len(''.join([x for x in set(row['question2']) if x not in stops])))
    X['char_diff_unique_stop'] = data.apply(char_diff_unique_stop, axis=1, raw=True)

    def char_ratio(row):
      l1 = len(''.join(row['question1'])) 
      l2 = len(''.join(row['question2']))
      if l2 == 0:
        return np.nan
      if l1 / l2:
        return l2 / l1
      else:
        return l1 / l2
    X['char_ratio'] = data.apply(char_dchar_ratioiff, axis=1, raw=True)

    return X