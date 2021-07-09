import json
import codecs
from PIL import Image, ImageDraw, ImageFont
import pyzbar.pyzbar as pyzbar
import serial  # 引用pySerial模組
from keras.models import Model
from keras.models import load_model
from skimage.transform import resize
import cv2
import sys
from annoy import AnnoyIndex
import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout, Lambda
from keras.models import Model, load_model
from bson.binary import Binary
import pickle
from pymongo import MongoClient
import random
import time
import numpy as np
import os
from datetime import datetime
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# for Model definition/training


# required for semi-hard triplet loss:


# Packages for approximate nearest neighbor

np.set_printoptions(threshold=sys.maxsize)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 設定Serial
ser = serial.Serial('COM4', 9600)


class Facenet():
    def __init__(self, inception_resnet_v1_path, img_size):
        self.img_size = img_size

        print(f'Loading facenet model...')
        start = time.time()
        model_path = os.path.abspath(inception_resnet_v1_path)
        inception_resnet_v1 = load_model(model_path, compile=False)

        # Remove the dropout layer, full-connected layer
        # and the batch-normalizing layer
        # from the original Inception ResNet-V1 model.
        new_input_layer = inception_resnet_v1.input
        new_output_layer = inception_resnet_v1.layers[-4].output
        self.model = Model(new_input_layer, new_output_layer)
        last = time.time()

        print(f'Loaded facenet model : {model_path}')
        self.model.summary()
        print(f'Time spent on loading model: {(last-start)} seconds')

    # Input an aligned image.
    # Output the embedding of the image
    def calc_embs(self, img):
        preprocessed_img = self.__prewhiten(img)
        embedding = self.model.predict_on_batch(preprocessed_img)
        #embedding = self.__l2_normalize(embedding)

        return embedding

    # Normalize the picture in preprocessing stage
    def __prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    # L2 normalize to produce embeddings
    def __l2_normalize(self, x, epsilon=1e-10):
        output = x / \
            np.sqrt(np.maximum(np.sum(np.square(x), keepdims=True), epsilon))
        return output


# This network defination must same as that be used in train-unit
def create_base_network(image_input_shape=(1792,), embedding_size=128):
    """
    Base network to be shared (eq. to feature extraction).
    """
    input_image = Input(shape=image_input_shape)

    #x = Flatten()(input_image)
    x = Dropout(0.1)(input_image)
    x = Dense(512, activation='sigmoid')(x)
    x = Dropout(0.1)(x)
    x = Dense(embedding_size)(x)
    x = Lambda(lambda x: nn.l2_normalize(x, axis=1, epsilon=1e-10))(x)

    base_network = Model(inputs=input_image, outputs=x)

    return base_network


def fetch_weight_from_mongodb(db):
    # Selection collection from database
    weight_collection = db['weight']

    bin_weight = weight_collection.find()

    if bin_weight.count(True) == 0:
        raise "The weight is not in the MnogoDB. Please traing first."
    bin_weight = bin_weight[0]['model']

    print("Weight loaded from MongoDB succesfully.")

    return pickle.loads(bin_weight)


def copy_weights(target, weights):
    for layer_target, weight in zip(target.layers, weights):
        layer_target.set_weights(weight)

    return target


def create_model_from_mongodb(db):
    base_model = create_base_network()

    # Fetch weights from MongoDB
    weights = fetch_weight_from_mongodb(db)

    # Copy weights to base model
    weighted_model = copy_weights(base_model, weights)

    return weighted_model


def construct_ann_from_mongodb(db):
    # Selection collection from database
    face_id_db_collection = db['face_id']

    # Prepare the mapping from ANN index to user_id
    # and the ANN instance.
    ann_map = {}
    embedding_size = 128
    ann = AnnoyIndex(embedding_size, 'euclidean')

    # Load all of the faces form MongoDB
    i = 0
    for user in face_id_db_collection.find():
        ann_map[i] = user['user_id']
        ann.add_item(i, user['embedding'])
        i += 1
    del i

    print("Loaded user_id and embeddings from MongoDB successfully.")
    print("Constructing ANN searching trees.")

    # Construct the ANN searching tree
    # Fixed to 10 trees.
    # [TODO] Adaptive tree number accroding to faces
    ann.build(10)

    print("ANN searching tree constructed.")

    return (ann_map, ann)


def face_recog_proc(faceId, db, model, ann_map, ann):

    print("Embedding calculated")

    # Perform ANN and send result to front-end-server
    # Perform ANN search. Get top 10 nearest neighbor.
    # If the following two condition is satisfied, return the user_id.
    # Otherwise, returns user not found.
    # Condition#1:
    #   There existed at least one neighbor that distance<=dist_thld.
    # Condition#2:
    #    The number of neighbors that satisfied cond#1 is greater than cnt_thld
    # If there existed mor than one lables satisfied the above two conditions,
    # return user not found.
    embedding = model.predict(faceId)[0]
    dist_thld = 0.8
    cnt_thld = 1
    ids, dists = ann.get_nns_by_vector(embedding, 10, include_distances=True)
    # Check which neighbors satisfied cond#1

    near_enough_neighbor = {}
    for id, dist in zip(ids, dists):
        user_id = ann_map[id]
        if dist <= dist_thld:
            # print('User ID: {} {}'.format(user_id, dist))
            if user_id in near_enough_neighbor.keys():
                near_enough_neighbor[user_id] += 1
            else:
                near_enough_neighbor[user_id] = 1

    # Check which neighbor satisfied cond#2
    result_cnt = 0
    result_user_id = -1
    for user_id in near_enough_neighbor:
        if near_enough_neighbor[user_id] > cnt_thld:
            result_cnt += 1
            result_user_id = user_id
    # If there are exately one lable satisfied the two conditions,
    # return it. Otherwise, return user not found.
    result_msg = {}
    if result_cnt == 1:
        result_msg = result_user_id
    else:
        result_msg = "NOTFOUND"

    return result_msg


def main():

    model_path = 'models/Inception_ResNet_v1_MS_Celeb_1M.h5'
    cascade_path = os.path.abspath('./models/haarcascade_frontalface_alt2.xml')
    print('I am a deep-unit')
    cascade = cv2.CascadeClassifier(cascade_path)
    print('Haar cascade classifier loaded.')

    facenet = Facenet(model_path, (160, 160))

    db_uri = "mongodb://localhost"
    try:
        db_client = MongoClient(db_uri)
        db_database = db_client['vireality_face_recog_backend']
        db_client.server_info()
        print("Connected to database")
    except:
        print("Connect to database failed. Check if mongoDB alive.")

    # Create model
    embedding_model = create_model_from_mongodb(db_database)

    # Construct the ANN searching tree
    ann_map, ann = construct_ann_from_mongodb(db_database)

    # cv2 video to img
    cap = cv2.VideoCapture(0)

    while True:
        qrcode_flag = False
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        barcodes = pyzbar.decode(gray)

        if (len(barcodes) != 0):
            barcode = barcodes[0]
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            barcodeData = barcode.data.decode("utf-8")
            print("---------qrcode get-----------")
            print(barcodeData)
            if (barcodeData == "visitor"):
                qrcode_flag = True

        if (not qrcode_flag):

            faces = cascade.detectMultiScale(frame,
                                             scaleFactor=1.1,
                                             minNeighbors=3)
            margin = 10
            # If there is no face found, reject it
            if len(faces) == 0:
                print("no face in the picture")
            else:
                # print(faces)

                # Crop and resize the image to fit the input shape of CNN
                (x, y, w, h) = faces[0]
                cropped_img = frame[max(y-margin//2, 0):y+h+margin//2,
                                    max(x-margin//2, 0):x+w+margin//2, :]

                aligned_img = resize(cropped_img, (160, 160), mode='reflect')

                # [DEBUG] Display aligned images
                disp_img = np.array(aligned_img, dtype=np.float32)
                disp_img = cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR)
                cv2.imshow('Aligned image', disp_img)
                cv2.waitKey(1)

                aligned_img = np.array([aligned_img])

                # Recognize the face
                face_id = facenet.calc_embs(aligned_img)
                result = face_id

                result_msg = face_recog_proc(
                    face_id,
                    db_database,
                    embedding_model,
                    ann_map,
                    ann)
                print("====== result_msg ======")
                print(result_msg)

                if(result_msg == 'ktony'):
                    print("open door")
                    try:
                        ser.write(b'open_door')
                        time.sleep(5)
                    except KeyboardInterrupt:
                        print('connect to arduino failed')

    # [DEBUG] Distroy window of image showing
    # 釋放攝影機
    cap.release()
    cv2.destroyAllWindows()
    ser.close()    # 清除序列通訊物件

    LOG_FORMAT = '%(asctime)s [recog-unit]: [%(levelname)s] %(message)s'
    print(level=print, format=LOG_FORMAT)

    print('I am a recog-unit')


if __name__ == "__main__":
    main()
