
import gensim
import nltk
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

def filter_NNP_from_chunk_tree(tree):
  stopwords = nltk.corpus.stopwords.words('english')
  result = []
  for subtree in tree:
    if type(subtree) != nltk.tree.Tree and \
      (subtree[0].lower() in stopwords or subtree[1] == '.'):
      continue
    
    if type(subtree) == nltk.tree.Tree:
      subtree_result = filter_NNP_from_chunk_tree(subtree)
      result.append(' '.join(subtree_result))
      continue
    
    print('not stopwords:', subtree[0])
    result.append(subtree[0])
  return result

sentence = 'Why did Microsoft choose core m3 and not core i3 home Surface Pro 4?'
#sentence = 'How does the Surface Pro himself 4 compare with iPad Pro?'
chunked_tokens = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
print(filter_NNP_from_chunk_tree(chunked_tokens))

model = gensim.models.word2vec(options.model_path)
print(model['iPad'])