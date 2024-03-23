FROM python:3
ADD requirements.txt /
RUN pip install -r requirements.txt
ADD train.py /
CMD [ "python", "./train.py" ]