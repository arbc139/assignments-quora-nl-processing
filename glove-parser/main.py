#!/usr/bin/python

import json
#import pymysql
import re
import sys

def parse_commands(argv):
  from optparse import OptionParser
  parser = OptionParser('"')
  parser.add_option('-i', '--inputFile', dest='input_file', help='Input glove file')
  parser.add_option('-d', '--dimension', type='int', dest='dimension', help='vector dimension')
  
  options, otherjunk = parser.parse_args(argv)
  return options

options = parse_commands(sys.argv[1:])

#db = pymysql.connect(host='localhost', user='root', password='', db='ai', charset='utf8')

def is_float(number):
  try:
    float(number)
  except ValueError:
    return False
  return True

rows = []
with open(options.input_file, 'r') as glove_file:
  while True:
    line = glove_file.readline()
    if not line:
      break
    line_arr = line.split()
    word_element_count = len(line_arr) - options.dimension
    word = ' '.join(line_arr[:word_element_count])
    vector = json.dumps(line_arr[word_element_count:])

    if len(line_arr[word_element_count:]) != options.dimension:
      raise RuntimeError('Vector dimension is not equal to parameter')

    rows.append([word, vector])
    print('word:', word)
    print('vector dimension:', len(line_arr[word_element_count:]))
"""
with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.executemany(
    'INSERT INTO GLOVE (word, vector) VALUES (%s, %s)',
    rows
  )
  db.commit()
"""