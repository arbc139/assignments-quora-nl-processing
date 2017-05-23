import sys, os
sys.path.append(os.path.abspath(".."))

import nltk

from custom_library.csv_manager import CsvReader, CsvWriter
from custom_library.utils import get_current_millis, get_elapsed_seconds

lemmatizer = nltk.stem.WordNetLemmatizer()

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('--testFile', dest='test_file')
  parser.add_option('--trainFile', dest='train_file')
  parser.add_option('--refineTestFile', dest='refine_test_file')
  parser.add_option('--refineTrainFile', dest='refine_train_file')

  options, otherjunk = parser.parse_args(argv)
  return options

options = parse_commands(sys.argv[1:])

def process_sentence(sentence):
  return filter_NNP_from_chunk_tree(
    nltk.corpus.stopwords.words('english'),
    nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
  )

def filter_NNP_from_chunk_tree(stopwords, tree):
  result = []
  for subtree in tree:
    if type(subtree) != nltk.tree.Tree and \
      (
        subtree[0].lower() in stopwords or \
        len(subtree[0]) < 2 or \
        subtree[1] == '.'
      ):
      continue
    
    if type(subtree) == nltk.tree.Tree:
      subtree_result = filter_NNP_from_chunk_tree(stopwords, subtree)
      result.append('_'.join(subtree_result))
      continue
    
    result.append(lemmatizer.lemmatize(subtree[0]))
  return result



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
refined_train_data = []
for train in train_data:
  refined_question1_words = process_sentence(train['question1'])
  refined_question2_words = process_sentence(train['question2'])
  refined_train_data.append({
    'id': train['id'],
    'qid1': train['qid1'],
    'qid2': train['qid2'],
    'question1': ' '.join(refined_question1_words),
    'question2': ' '.join(refined_question2_words),
    'is_duplicate': train['is_duplicate'],
  })
print('Refine train data time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
refined_test_data = []
for test in test_data:
  refined_question1_words = process_sentence(test['question1'])
  refined_question2_words = process_sentence(test['question2'])
  refined_test_data.append({
    'test_id': test['test_id'],
    'question1': ' '.join(refined_question1_words),
    'question2': ' '.join(refined_question2_words),
  })
print('Refine test data time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
with open(options.refine_train_file, 'w+') as refine_train_file:
  csv_writer = CsvWriter(refine_train_file, ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'])
  csv_writer.write_header()
  for train in refined_train_data:
    csv_writer.write_row(train)
print('Write refine train data file time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))

elapsed_millis = get_current_millis()
with open(options.refine_train_file, 'w+') as refine_test_file:
  csv_writer = CsvWriter(refine_train_file, ['test_id', 'question1', 'question2'])
  csv_writer.write_header()
  for test in refined_test_data:
    csv_writer.write_row(test)
print('Write refine test data file time:', get_elapsed_seconds(get_current_millis(), elapsed_millis))
