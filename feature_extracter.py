
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
    
    # Feature1. Shard word match feature.
    X['word_match'] = data.apply(shared_word_match, axis=1, raw=True)

    def tfidf_word_match_weight(row):
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
    
    # Feature2. tf-idf word_weight feature.
    X['tfidf_word_match'] = data.apply(tfidf_word_match_weight, axis=1, raw=True)

    def word_vector_similarity(row):
      question1_vector = np.zeros(300)
      for word in row['question1']:
        question1_vector = np.add(question1_vector, get_word_vector(self.word_vector_model, word))
      normalized_question1_vector = normalize_vector(question1_vector)
      
      question2_vector = np.zeros(300)
      for word in row['question2']:
        question2_vector = np.add(question2_vector, get_word_vector(self.word_vector_model, word))
      normalized_question2_vector = normalize_vector(question2_vector)
      
      return cosine_distance(sentence1_vector, sentence2_vector)

    # Feature3. word vector similarity
    X['word_vector_similarity'] = refined_data.apply(word_vector_similarity, axis=1, raw=True)

    return X