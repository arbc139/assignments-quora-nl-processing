#!/usr/bin/python

import json
import pymysql

db = pymysql.connect(host='localhost', user='root', password='', db='ai', charset='utf8')

rows = []
with open('../dataset/glove.840B.300d.txt', 'r') as glove_file:
  while True:
    line = glove_file.readline()
    if not line:
      break
    line_arr = line.split()
    word = line_arr[0]
    vector = json.dumps(line_arr[1:])
    rows.append([word, vector])

with db.cursor(pymysql.cursors.DictCursor) as cursor:
  cursor.executemany(
    'INSERT INTO GLOVE (name, vector) VALUES (%s, %s)',
    rows
  )
  db.commit()
