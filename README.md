# 2017-1학기 AI Quora Project

## Docker
이 repository는 docker를 지원하고 있습니다.

### Build
`$ docker build -t totoro/quora-nl-processing .`

### Run
`$ docker run -d totoro/quora-nl-processing`

### Connect shell 
`$ docker exec -it totoro/quora-nl-processing /bin/bash`

### Trouble shooting
Build를 할 때, storage가 모자라는 에러가 뜨면 다음과 같이 해결하면 된다.

```
$ echo '{"storage-driver":"overlay2"}' > /etc/docker/daemon.json 
$ systemctl restart docker
$ docker info
```

이때 Storage type이 overlag2로 변경되어있으면 성공이다. 이후 docker build를 재개하면 된다.