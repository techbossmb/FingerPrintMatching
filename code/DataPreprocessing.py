import os
from glob import glob
import random
import string
import pandas as pd
import numpy as np
import pickle

from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from keras.models import Model
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

class DataPreprocessing:

    def __init__(self, datadir, featureextraction_modelfile, isTest=True):
        # isTest - flag used to decide current instance is training or test. Used to overwrite existing fingerprint db file

        self.datadir = datadir
        # right middlefinger first -> mod 4 = 0
        self.tags = ['right_middlefinger', 'left_forefinger', 'right_forefinger', 'left_middlefinger']

        self.personidentifiermapfile = '..{0}data{0}personidentifiermapfile.pickle'.format(os.sep)
        if os.path.exists(self.personidentifiermapfile) and isTest:
            self.load_personIdentifierMap()
        else:
            self.person_guid_map = {}

        self.fingerprintDBfile = '..{0}data{0}fingerprintdb.pickle'.format(os.sep)
        if os.path.exists(self.fingerprintDBfile) and isTest:
            self.load_fingerprintDB()
        else:
            self.fingerprintDatabase = {}

        print('loading resnet model for image feature extraction')
        resnet_model = load_model(featureextraction_modelfile)
        self.resnet_feature_extractor =  Model(inputs=resnet_model.input, outputs=resnet_model.get_layer('global_average_pooling2d_1').output)

    def getDataFrameFromImage(self):
        fingerprints = []
        imagefiles = glob('{}{}*.tif'.format(self.datadir, os.sep))
        for imagefile in imagefiles:
            person_id, finger_id = imagefile.split(os.sep)[-1].split('.')[0].split('aug')[0].split('_')
            if person_id in self.person_guid_map:
                guid = self.person_guid_map[person_id]
            else:
                guid = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
                self.person_guid_map[person_id] = guid

            tag = self.tags[int(finger_id)%4]
            
            fingerprint = {
                'file_id': imagefile.split(os.sep)[-1],
                'person_id': guid,
                'finger_id': tag     
            }
            fingerprints.append(fingerprint)
        
        with open(self.personidentifiermapfile, 'wb') as handle:
            pickle.dump(self.person_guid_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df = pd.DataFrame.from_dict(fingerprints)
        df['class'] = df['person_id'] + df['finger_id']
        df = df.sample(frac=1, random_state=1337)
        return df

    def load_fingerprintDB(self):
        print('loading fingerprint db')
        if os.path.exists(self.fingerprintDBfile):
            with open(self.fingerprintDBfile, 'rb') as handle:
                self.fingerprintDatabase = pickle.load(handle)
        else:
            raise Exception("Fingerprint DB file does not exist. Train fingerprint model first to build a fingerprint DB")

    def load_personIdentifierMap(self):
        print('loading person identifier map from file')
        if os.path.exists(self.personidentifiermapfile):
            with open(self.personidentifiermapfile, 'rb') as handle:
                self.person_guid_map = pickle.load(handle)
        else:
            raise Exception("Fingerprint DB file does not exist. Train fingerprint model first to build a fingerprint DB")
            
            
    def _getImageFeatures(self, df):
        features = []
        for i in range(df.shape[0]):
            features.append(img_to_array(load_img('{}{}{}'.format(self.datadir, os.sep, df.file_id.iloc[i])))/255.0)
        features = np.asarray(features)
        return features

    def _getResnetFeatures(self, df, modelfilepath):
        
        print('generating resnet features')
        imagefeatures = self._getImageFeatures(df)
        resnet_features = self.resnet_feature_extractor.predict(imagefeatures)
        return resnet_features

    def extractImageFeaturesToSiameseDataFrame(self, df, featureextractor_modelfile):
        resnet_features = self._getResnetFeatures(df, featureextractor_modelfile)
        print('building fingerprint-feature DB')
        self.fingerprintDatabase = dict(zip(df.file_id.values, resnet_features))
        with open(self.fingerprintDBfile, 'wb') as handle:
            pickle.dump(self.fingerprintDatabase, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('creating new dataset from generated resnet features')
        siamesedataset_dict = []
        for i in range(df.shape[0]):
            for j in range(df.shape[0]):
                siamese_dict = {
                    'image_one': df.file_id.iloc[i],
                    'image_two': df.file_id.iloc[j],
                    'similarity': 1 if df['class'].iloc[i]==df['class'].iloc[j] else 0,
                    'resnetfeatures_one': resnet_features[i],
                    'resnetfeatures_two': resnet_features[j]
                }
                siamesedataset_dict.append(siamese_dict)
        siamese_df = pd.DataFrame.from_dict(siamesedataset_dict)
        return siamese_df

    def getSiameseDataset(self, siamese_df):
        print('creating siamese network dataset')
        feature_one = []
        feature_two = []
        label = []
        size = siamese_df.shape[0]
        for i in range(size):
            feature_one.append(siamese_df.resnetfeatures_one.iloc[i])
            feature_two.append(siamese_df.resnetfeatures_two.iloc[i])
            label.append(siamese_df.similarity.iloc[i])
            if i % 100000 == 0: print('current index {}'.format(i))               
        feature_one = np.asarray(feature_one)
        feature_two = np.asarray(feature_two)
        label = np.asarray(label)
        return feature_one, feature_two, label

    def createTestData(self, fingerprint_to_match):
        imagefeature_one = img_to_array(load_img(fingerprint_to_match))/255.0

        print('creating fingerprint features')
        feature_one = self.resnet_feature_extractor.predict(np.asarray([imagefeature_one,]))

        dbsize = len(self.fingerprintDatabase)
        siamesefeature_one = np.asarray([feature_one] * dbsize)
        siamesefeature_one = siamesefeature_one.squeeze()
        siamesefeature_two = np.asarray(list(self.fingerprintDatabase.values()))
        return siamesefeature_one, siamesefeature_two

    def createTestDataWithPersonID(self, fingerprint_to_match, person_id, template_DB_keylist):
        ''' @bugfix: template_DB_keylist argu should be supplied as an empty list when passing person_id '''
        
        # get key from value => get corresponding tif file from person id
        tifprefix = None
        for key, value in self.person_guid_map.items():
            if value == person_id:
                tifprefix = key
                break
        if tifprefix is None:
            raise ValueError("Person ID not found in DB")
        
        # get all fingerprint DB values for the given tifprefix
        templates_to_match = []
        for key, value in self.fingerprintDatabase.items():
            if key.startswith('{}_'.format(tifprefix)):
                templates_to_match.append(value)
                template_DB_keylist.append(key)
        siamesefeature_two = np.asarray(templates_to_match)

        imagefeature_one = img_to_array(load_img(fingerprint_to_match))/255.0
        print('creating fingerprint features')
        feature_one = self.resnet_feature_extractor.predict(np.asarray([imagefeature_one,]))
        dbsize = siamesefeature_two.shape[0]
        siamesefeature_one = np.asarray([feature_one] * dbsize)
        siamesefeature_one = siamesefeature_one.squeeze()
        return siamesefeature_one, siamesefeature_two

        
