import os
import random
import math
import numpy as np
import string
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight

from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, multiply, Lambda, add, concatenate, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras.callbacks import TensorBoard


class FingerprintModel:
    def __init__(self, fingerprint_modelpath, featureextractor_modelpath):
        self.siamese_modelpath = fingerprint_modelpath
        self.resnet_modelpath = featureextractor_modelpath
        if os.path.exists(fingerprint_modelpath):
            print('loading fingerprint model from file')
            self.siamese_model = load_model(fingerprint_modelpath)
        else:
            self.siamese_model = None
        
    def train(self, modeltype, predictor, response, val_predictor=None, val_response=None):

        if modeltype == 'ResNet50':
            base_model = ResNet50(weights='imagenet', input_shape=(300,300,3), include_top=False)
            for layer in base_model.layers:
                layer.trainable = True
            softmaxlayer = base_model.output
            softmaxlayer = GlobalAveragePooling2D()(softmaxlayer)
            softmaxlayer = Dense(2048, activation='relu')(softmaxlayer)
            softmaxlayer = Dense(512, activation='relu')(softmaxlayer)
            softmaxlayer = Dense(40, activation='softmax')(softmaxlayer)

            model = Model(inputs=base_model.input, outputs=softmaxlayer)
            print(model.summary())
            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
            model.fit(predictor, response, batch_size=1, epochs=100)
            model.save(self.resnet_modelpath)

        elif modeltype == 'Siamese':
            feature_one = Input(shape=(2048,))
            feature_two = Input(shape=(2048,))
            dot_product = multiply([feature_one, feature_two])
            minus_duplicate = Lambda(lambda x: -x)(feature_two)
            diff = add([feature_one, minus_duplicate])
            diff_squared = multiply([diff, diff])
            features = [feature_one, feature_two, dot_product, diff_squared]
            feature_layer = concatenate(features)
            #dense_layer = Dropout(0.7)(feature_layer)
            sum_of_features = sum(feature.shape[1].value for feature in features)
            layer_size = int(math.floor(math.sqrt(sum_of_features)))*4
            dense_layer = Dense(layer_size, activation='relu')(feature_layer)
            #dense_layer = Dropout(0.7)(dense_layer)
            dense_layer = Dense(int(layer_size/2), activation='relu')(dense_layer)
            #dense_layer = Dropout(0.6)(dense_layer)
            softmax = Dense(2, activation='softmax', name='softmax')(dense_layer)

            model = Model([feature_one, feature_two], softmax, name='fingermatch')
            print(model.summary())
            model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

            # address class imbalance using weighted loss function
            tr_onehotlabels_int = np.argmax(response, axis=1)
            classweight = class_weight.compute_class_weight('balanced', np.unique(tr_onehotlabels_int), tr_onehotlabels_int)
            # generate unique identified for tensorboard event log
            tensorboard_path = 'SiameseNetwork_{}'.format(''.join(random.choice(string.ascii_lowercase) for i in range(5)))

            model.fit(predictor, response,
                    validation_data=(val_predictor, val_response),
                    batch_size=32, epochs=100, 
                    class_weight=classweight,
                    callbacks=[TensorBoard(log_dir='..{0}logs{0}{1}'.format(os.sep, tensorboard_path))])

            model.save(self.siamese_modelpath)
            self.siamese_model = model

    def evaluate(self, valfeatures_one, valfeatures_two, vallabels):
        match_prediction = self.siamese_model.predict([valfeatures_one, valfeatures_two])
        match_prediction = np.argmax(match_prediction, axis=1)

        print('number of positive predictions = {}\n total number of actual positives = {}'
            .format(sum(match_prediction), sum(vallabels)))

        tp,fp,tn,fn,tnr,fnr = 0.0,0.0,0.0,0.0,0.0,0.0
 
        for i in range(len(match_prediction)):
            if match_prediction[i] == 1:
                if vallabels[i] == 1:
                    tp += 1
                elif vallabels[i] == 0:
                    fp += 1
            elif match_prediction[i] == 0:
                if vallabels[i] == 0:
                    tn += 1
                elif vallabels[i] == 1:
                    fn += 1
        print('TP: {}, FP: {}, TN: {}. FN: {}'.format(tp, fp, tn, fn))
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        tnr = tn/(fp+tn)
        fnr = fn/(tp+fn)
        print('Accuracy: {}'.format((tp+tn)/len(match_prediction)))
        print('Misclassification: {}'.format((fp+fn)/len(match_prediction)))
        print('TPR: {}'.format(tpr))
        print('FPR: {}'.format(fpr))
        print('Precision : {}'.format(tp/(fp+tp)))

    def predict(self, feature_one, feature_two):
        match_predictions = self.siamese_model.predict([feature_one, feature_two])
        match_index = np.argmax(match_predictions[:,1])
        match_prob = match_predictions[match_index,1]
        return match_index, match_prob
