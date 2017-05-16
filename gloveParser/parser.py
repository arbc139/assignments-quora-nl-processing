#!/usr/bin/python

#import json
#import pymysql
import re

"""
def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('-i', '--inputFile', dest='input_file', help='Input glove file')
  parser.add_option('-d', '--dimension', type='int', dest='dimension', help='vector dimension')
  
  options, otherjunk = parser.parse_args(argv)
  return options
"""

#db = pymysql.connect(host='localhost', user='root', password='', db='ai', charset='utf8')

def is_utf8(word):
  return isinstance(word, str) or isinstance(word, unicode)

def get_word2vec(options):
  word2vec = {}
  with open(options['input_file'], 'r') as glove_file:
    while True:
      line = glove_file.readline()
      if not line:
        break
      line_arr = line.split()
      word_element_count = len(line_arr) - options['dimension']
      word = ' '.join(line_arr[:word_element_count])
      vector = line_arr[word_element_count:]

      if len(vector) != options['dimension']:
        raise RuntimeError('Vector dimension is not equal to parameter')
      """
      if not is_utf8(word):
        print('word is not utf8:', word)
        continue
      """

      word2vec[word] = vector
  return word2vec[word]

"""
with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.executemany(
    'INSERT INTO GLOVE (word, vector) VALUES (%s, %s)',
    rows
  )
db.commit()
"""