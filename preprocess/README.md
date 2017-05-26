## Refine word
`python refine_word.py --trainFile ../dataset/input/quora/train.csv --testFile ../dataset/input/quora/test.csv --refineTrainFile ../dataset/output/refined_train.csv --refineTestFile ../dataset/output/refined_test.csv`

`python refine_word.py --trainFile ../dataset/input/quora/sample_train.csv --testFile ../dataset/input/quora/sample_test.csv --refineTrainFile ../dataset/output/sample_refined_train.csv --refineTestFile ../dataset/output/sample_refined_test.csv`

## Save word vector
`nohup python save_word_vector.py --rawVectorFile ../dataset/input/google/GoogleNews-vectors-negative300.bin --wordVectorFile ../processed_word_vector.gensim > log_save_word_vector.out &`