import json
import pandas as pd
import helpers
import logging
import requests

from io import BytesIO
from flask import Flask
from flask import request
from logging.handlers import TimedRotatingFileHandler


model = load_model('model.json', 'model.h5')
airplane = pd.read_csv('airplane.csv')
bicycle = pd.read_csv('bicycle.csv')
car = pd.read_csv('car.csv')
motorcycle = pd.read_csv('motorcycle.csv')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = TimedRotatingFileHandler('test.log', when='d', interval=1, backupCount=3)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

app = Flask(__name__)

@app.route('/')
def hello_world():
    logger.debug('Default route')
    return app.send_static_file('form.html')

@app.route('/read', methods=['GET'])
def read(tag):
    
    if tag == 'airplane':
        urls = list(airplane['urls'])
        r, g, b = airplane.r.mean(), airplane.g.mane(), airplane.b.mean()
    if tag == 'bicycle':
        urls = list(bicycle['urls'])
        r, g, b = bicycle.r.mean(), bicycle.g.mane(), bicycle.b.mean()
    if tag == 'car':
        urls = list(car['urls'])
        r, g, b = car.r.mean(), car.g.mane(), car.b.mean()
    if tag == 'motorcycle':
        urls = list(motorcycle['urls'])
        r, g, b = motorcycle.r.mean(), motorcycle.g.mane(), motorcycle.b.mean()
        
    my_dict = {'rgb': (r, g, b), 'urls': urls}

    return json.dumps(my_dict)

@app.route('/predict', methods=['POST'])
def predict():
    logger.debug('Predict route called')
    
    # only url as input
    url = request.form['url']

    logger.debug('Received the following params:' + str(name) + ' and ' + str(age))
    
    # load image from url
    response = requests.get(urls[i])
    image = Image.open(BytesIO(response.content))
    
    # process image
    # predict class
    result = helpers.predict2(image)
    
    idx = result.index(1)
    if idx == 0:
        tag = 'airplane'
    elif idx == 1:
        tag = 'bicycle'
    elif idx == 2:
        tag = 'car'
    else:
        tag = 'motorcycle'
    
    # get giphy from class
    url_giphy = helpers.get_giphy(tag)
    
    my_dict = {'url_giphy': url_giphy}

    return json.dumps(my_dict)

if __name__ == '__main__':
    app.run()