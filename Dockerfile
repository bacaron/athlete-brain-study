FROM ubuntu:18.04

MAINTAINER Brad Caron <bacaron@iu.edu>

RUN apt-get update && apt-get install -y python3.6 python3-pip jq git

RUN pip3 install numpy pandas matplotlib sklearn seaborn statsmodels scipy pinguoin

RUN ldconfig && mkdir -p /N/u /N/home /N/dc2 /N/soft

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
