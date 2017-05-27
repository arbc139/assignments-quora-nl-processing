
import numpy as np
import time

def get_current_millis():
  return int(round(time.time() * 1000))

def get_elapsed_seconds(current_time, elapsed_millis):
  return (current_time - elapsed_millis) / 1000.0

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
