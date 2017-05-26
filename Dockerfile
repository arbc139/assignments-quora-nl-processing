FROM parrotprediction/course-xgboost
MAINTAINER Do yeong Kim <arbc139@gmail.com>

# pip upgrade
RUN pip install --upgrade pip

# Install my works
RUN git clone https://github.com/arbc139/quora-nl-processing
RUN cd quora-nl-processing; pip install -r requirements.txt
RUN python -m nltk.downloader all
RUN cd ~

ENTRYPOINT cd quora-nl-processing; git pull --rebase; pip install -r requirements.txt; cd ~