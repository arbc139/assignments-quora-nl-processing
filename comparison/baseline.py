import sys, os
sys.path.append(os.path.abspath(".."))

from custom_library.csv_manager import CsvReader, CsvWriter

test_data = None
with open('../dataset/input/quora/test.csv', 'r') as test_file:
  csv_reader = CsvReader(test_file)
  test_data = csv_reader.get_dict_list_data()

with open('baseline_submission.csv', 'w+') as submission_file:
  csv_writer = CsvWriter(submission_file, ['test_id', 'is_duplicate'])
  csv_writer.write_header()
  for test in test_data:
    csv_writer.write_row({
      'test_id': test['test_id'],
      'is_duplicate': 0,
    })