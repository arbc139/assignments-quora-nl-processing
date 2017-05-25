FROM parrotprediction/course-xgboost
MAINTAINER Do yeong Kim <arbc139@gmail.com>

# Install zsh
RUN apt-get install zsh
RUN sudo chsh -s `which zsh`

# pip upgrade
RUN pip install --upgrade pip

# Install my works
RUN git clone https://github.com/arbc139/quora-nl-processing
RUN cd quora-nl-processing; pip install -r requirements.txt
RUN python -m nltk.downloader all
