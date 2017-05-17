#!/usr/bin/python

import json
import pymysql
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

db = pymysql.connect(host='localhost', user='root', password='', db='ai', charset='utf8')

def is_float(number):
  try:
    float(number)
  except ValueError:
    return False
  return True

def is_utf8(word):
  return isinstance(word, str) or isinstance(word, unicode)

words = []
vectors = []
with open(options.input_file, 'r') as glove_file:
  while True:
    line = glove_file.readline()
    if not line:
      break
    line_arr = line.split()
    word_element_count = len(line_arr) - options.dimension
    word = ' '.join(line_arr[:word_element_count])
    vector = line_arr[word_element_count:]

    if len(vector) != options.dimension:
      raise RuntimeError('Vector dimension is not equal to parameter')
    
    if not is_utf8(word):
      print('word is not utf8:', word)
      continue

    words.append(word)
    vector = [re.sub('\'', '', value) for value in vector]
    vectors.append(vector)
    print('word:', word)
    print('vector dimension:', len(line_arr[word_element_count:]))

if len(words) != len(vectors):
  raise RuntimeError('Count of words is not equal to count of vectors')

with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.executemany(
    'INSERT INTO VECTOR (attr0, attr1, attr2, attr3, attr4, attr5, attr6, attr7, attr8, attr9, attr10, attr11, attr12, attr13, attr14, attr15, attr16, attr17, attr18, attr19, attr20, attr21, attr22, attr23, attr24, attr25, attr26, attr27, attr28, attr29, attr30, attr31, attr32, attr33, attr34, attr35, attr36, attr37, attr38, attr39, attr40, attr41, attr42, attr43, attr44, attr45, attr46, attr47, attr48, attr49, attr50, attr51, attr52, attr53, attr54, attr55, attr56, attr57, attr58, attr59, attr60, attr61, attr62, attr63, attr64, attr65, attr66, attr67, attr68, attr69, attr70, attr71, attr72, attr73, attr74, attr75, attr76, attr77, attr78, attr79, attr80, attr81, attr82, attr83, attr84, attr85, attr86, attr87, attr88, attr89, attr90, attr91, attr92, attr93, attr94, attr95, attr96, attr97, attr98, attr99, attr100, attr101, attr102, attr103, attr104, attr105, attr106, attr107, attr108, attr109, attr110, attr111, attr112, attr113, attr114, attr115, attr116, attr117, attr118, attr119, attr120, attr121, attr122, attr123, attr124, attr125, attr126, attr127, attr128, attr129, attr130, attr131, attr132, attr133, attr134, attr135, attr136, attr137, attr138, attr139, attr140, attr141, attr142, attr143, attr144, attr145, attr146, attr147, attr148, attr149, attr150, attr151, attr152, attr153, attr154, attr155, attr156, attr157, attr158, attr159, attr160, attr161, attr162, attr163, attr164, attr165, attr166, attr167, attr168, attr169, attr170, attr171, attr172, attr173, attr174, attr175, attr176, attr177, attr178, attr179, attr180, attr181, attr182, attr183, attr184, attr185, attr186, attr187, attr188, attr189, attr190, attr191, attr192, attr193, attr194, attr195, attr196, attr197, attr198, attr199, attr200, attr201, attr202, attr203, attr204, attr205, attr206, attr207, attr208, attr209, attr210, attr211, attr212, attr213, attr214, attr215, attr216, attr217, attr218, attr219, attr220, attr221, attr222, attr223, attr224, attr225, attr226, attr227, attr228, attr229, attr230, attr231, attr232, attr233, attr234, attr235, attr236, attr237, attr238, attr239, attr240, attr241, attr242, attr243, attr244, attr245, attr246, attr247, attr248, attr249, attr250, attr251, attr252, attr253, attr254, attr255, attr256, attr257, attr258, attr259, attr260, attr261, attr262, attr263, attr264, attr265, attr266, attr267, attr268, attr269, attr270, attr271, attr272, attr273, attr274, attr275, attr276, attr277, attr278, attr279, attr280, attr281, attr282, attr283, attr284, attr285, attr286, attr287, attr288, attr289, attr290, attr291, attr292, attr293, attr294, attr295, attr296, attr297, attr298, attr299) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',
    vectors
  )
db.commit()

vector_ids = None
with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.execute('SELECT id FROM VECTOR ORDER BY id')
  vector_ids = cursor.fetchall()

if len(word) != len(vector_ids):
  raise RuntimeError('Count of words is not equal to count of vector_rows')

rows = []
for i in range(0, len(word)):
  rows.append([word[i], vector_ids[i]])

with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.executemany(
    'INSERT INTO WORD2VEC (word, vector_id) VALUES (%s, %s)',
    rows
  )
db.commit()
