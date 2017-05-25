
import time

def get_current_millis():
  return int(round(time.time() * 1000))

def get_elapsed_seconds(current_time, elapsed_millis):
  return (current_time - elapsed_millis) / 1000.0

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('--testFile', dest='test_file')
  parser.add_option('--trainFile', dest='train_file')
  parser.add_option('--submissionFile', dest='submission_file')

  options, otherjunk = parser.parse_args(argv)
  return options