FROM parrotprediction/course-xgboost
MAINTAINER Do yeong Kim <arbc139@gmail.com>

# Run upgrades
RUN echo "deb http://archive.ubuntu.com/ubuntu precise main universe" > /etc/apt/sources.list
RUN apt-get update

# Install my works
RUN git clone https://github.com/arbc139/quora-nl-processing
RUN cd quora-nl-processing; pip install -r requirements.txt
RUN python -m nltk.downloader all

EXPOSE 80
CMD ["/usr/sbin/apache2", "-D", "FOREGROUND"]