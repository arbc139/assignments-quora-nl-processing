import sys, os
sys.path.append(os.path.abspath(".."))

import gensim
from time_logger import TimeLogger

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  # Input file path
  parser.add_option('-i', '--wordVectorFile', dest='word_vector_file')

  options, otherjunk = parser.parse_args(argv)
  return options

arguments = parse_commands(sys.argv[1:])
time_logger = TimeLogger()

time_logger.start()
model = gensim.models.KeyedVectors.load(arguments.word_vector_file)
time_logger.log_with_elapse('Load word vector model time:')