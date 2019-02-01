# import packages
import numpy as np
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt

import time
import giphy_client
from giphy_client.rest import ApiException

from PIL import Image
from io import BytesIO

# imports for CNN
import seaborn as sns
import tensorflow as tf

import matplotlib.pyplot as plt
from quiver_engine import server 

from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import np_utils
from keras.utils import plot_model

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# define CONSTANTS
FLICKR_PUBLIC_KEY = 'edda29f5095149a8d093c14c14767765'
FLICKR_PRIVATE_KEY = '3ff6961e69c5c7d0'
GIPHY_API_KEY = 'xx6bu5kNzpAAlCZaqc2f4G46c3891Lkt'

def get_public_key():
    return FLICKR_PUBLIC_KEY

def get_private_key():
    return FLICKR_PRIVATE_KEY

def get_giphy_key():
    return GIPHY_API_KEY

def get_aveage_rgb(image):
    # convert image to a numpy array
    img = np.array(image)
    
    # get the shape of the image
    x, y, channel = img.shape
    
    # flatten the image
    img.shape = (x*y, channel)
    
    # return the average rgb
    return np.mean(img, axis=0)

def get_images(flic, tag, page, per_page):
    urls = []
    rgbs = [] 
    if not os.path.exists(str(tag)):
        os.makedirs(str(tag))
        
    for i in range(page):
        images = flic.photos.search(text=tag, per_page=per_page, extras='url_n', orientation='square', page=i)
        for j in range(len(images['photos']['photo'])):
            try:
                urls.append(images['photos']['photo'][j]['url_n'])
            except:
                pass
            
    for i in range(len(urls)):
        response = requests.get(urls[i])
        image = Image.open(BytesIO(response.content))
        rgbs.append(get_aveage_rgb(image))
        uri = str(tag) + '/' + str(tag) + str(i) + '.jpg'
        image.save(uri)
    return urls, rgbs

def filter_images(elements, uri, df):
    for i in elements:
            uri = str(tag) + '/' + str(tag) + str(i) + '.jpg'
            os.remove(uri)
            
def save_csv(tags, urls, rgbs):
    tags = [tags] * len(urls)
    data = {'tags': tags, 'rgbs': rgbs, 'urls': urls}
    df = pd.DataFrame(data)
    filename = tags[0] + '.csv'
    df.to_csv(filename)
    
def define_model(optimizer='adam', loss='categorical_crossentropy', input_shape=(200,200,3), metrics=['accuracy'], classes=0):
    model = Sequential()
    model.add(Conv2D(32,(3,3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(units = 256, activation='relu'))
    model.add(Dense(units = classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    return model

def load_data(path, target_size, batch_size, color_mode):
    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True, vertical_flip=True, 
                                       validation_split=0.3)
    training_data = datagen.flow_from_directory(path, target_size=target_size, batch_size=16, class_mode='categorical', 
                                                  subset='training', color_mode=color_mode, shuffle=True,
                                                  seed=42)
    validation_data = datagen.flow_from_directory(path, target_size=target_size, batch_size=batch_size,
                                                        class_mode='categorical', subset='validation', color_mode=color_mode, 
                                                        shuffle=True, seed=42)
    return training_data, validation_data

def train_model(model, training_data, validation_data, train_steps, epochs, val_steps):
    model.fit_generator(training_data, steps_per_epoch=train_steps, epochs=epochs, 
                    validation_data=validation_data, validation_steps=val_steps, verbose=1)
    return model

def load_model(path_to_json, path_to_weights, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    json_file = open(path_to_json, 'r')
    json_model = json_file.read()
    json.close()
    model = model_from_json(json_model)
    model.load_weights(path_to_weights)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model 

def predict(model, path_to_image, target_size, grayscale=False):
    test_img = image.load_img(path_to_image, target_size=target_size, grayscale=grayscale)
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return result

def save_model(model, name='model'):
    plot_model(model, to_file=name+'.png')
    model_json = model.to_json()
    with open(name+'.json', 'w') as file:
        file.write(model_json)
    model.save_weights(name+'.h5')
    
def evaluate_model(model, evaluation_data, steps=100):
    loss, accuracy = model.evaluate_generator(testing_data, steps=steps)
    print(loss, accuracy)
    
def predict2(model, image, target_size, grayscale=False):
    test_img = image.img_to_array(image)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)
    return result

def get_giphy(tag):


    # create an instance of the API class
    api_instance = giphy_client.DefaultApi()
    api_key = get_giphy_key() # str | Giphy API Key.
    q = tag 
    limit = 25 
    offset = 0 
    rating = 'g' 
    lang = 'en'
    fmt = 'json'

    try: 
        # Search Endpoint
        api_response = api_instance.gifs_search_get(api_key, q, limit=limit, offset=offset, rating=rating, lang=lang, fmt=fmt);
    except ApiException as e:
        print("Exception when calling DefaultApi->gifs_search_get: %s\n" % e)
        
    data = api_response.to_dict()
    return data['data'][0]['url']