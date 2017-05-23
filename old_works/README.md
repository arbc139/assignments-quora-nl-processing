python main.py --trainFile dataset/sample_train.csv --testFile dataset/sample_test.csv -o ../dataset/quora_data/submission.csv -m dataset/GoogleNews-vectors-negative300.bin

python main.py --trainFile ../dataset/quora_data/sample_train.csv --testFile ../dataset/quora_data/sample_test.csv -o ../dataset/quora_data/submission.csv -m ../dataset/google_vector_input/GoogleNews-vectors-negative300.bin

nohup python main.py --trainFile ../dataset/quora_data/sample_train.csv --testFile ../dataset/quora_data/sample_test.csv -o ../dataset/quora_data/submission.csv -m ../dataset/google_vector_input/GoogleNews-vectors-negative300.bin &

nohup python main.py --trainFile ../dataset/quora_data/train.csv --testFile ../dataset/quora_data/test.csv -o ../dataset/quora_data/submission.csv -m ../dataset/google_vector_input/GoogleNews-vectors-negative300.bin &

