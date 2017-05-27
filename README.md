# 2017-1학기 AI Quora Project

## Docker
이 repository는 docker를 지원하고 있습니다.

### Build
`$ docker build -t totoro/quora-nl-processing .`

### Run
`$ docker run -d -p 8888:8888 --name=quora-nl-processing -v ~/quora-nl-processing/dataset:/home/jovyan/work/quora-nl-processing/dataset totoro/quora-nl-processing`

### Connect shell 
`$ docker exec -it quora-nl-processing /bin/bash`

### Copy file from container to host
`$ docker cp <containerId>:/home/jovyan/work/quora-nl-processing/submission.csv ~/quora-nl-processing/dataset/output`

### Trouble shooting
Build를 할 때, storage가 모자라는 에러가 뜨면 다음과 같이 해결하면 된다.

```
$ echo '{"storage-driver":"overlay2"}' > /etc/docker/daemon.json 
$ systemctl restart docker
$ docker info
```

이때 Storage type이 overlag2로 변경되어있으면 성공이다. 이후 docker build를 재개하면 된다.

## Project
프로젝트를 실행하는 방법은 다음과 같다.
`python main.py --trainFile dataset/input/quora/train.csv --testFile dataset/input/quora/test.csv --refinedTrainFile dataset/input/quora/refined_train.csv --refinedTestFile dataset/input/quora/refined_test.csv --wordVectorFile processed_word_vector.gensim --submissionFile submission.csv`

Background에서 실행시키려면 다음과 같이 nohup을 이용하면 된다.
`nohup python main.py --trainFile dataset/input/quora/train.csv --testFile dataset/input/quora/test.csv --refinedTrainFile dataset/input/quora/refined_train.csv --refinedTestFile dataset/input/quora/refined_test.csv --wordVectorFile processed_word_vector.gensim --submissionFile submission.csv &`

Docker로 바로 실행시키려면 다음과 같이 실행시키면 된다.
`nohup docker exec -it quora-nl-processing (python /home/jovyan/work/quora-nl-processing/main.py --trainFile dataset/input/quora/train.csv --testFile dataset/input/quora/test.csv --refinedTrainFile dataset/input/quora/refined_train.csv --refinedTestFile dataset/input/quora/refined_test.csv --wordVectorFile processed_word_vector.gensim --submissionFile submission.csv > running_log.out) &`