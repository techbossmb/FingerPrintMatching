import os
from ImageAugmentation import ImageAugmentation
from DataPreprocessing import DataPreprocessing
from FingerprintModel import FingerprintModel

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

class  ModelBuilder:
    def __init__(self, imagedir):  
        augmenteddatadir = '..{0}data{0}images'.format(os.sep)
        resnetmodel_path = '..{0}model{0}resnet_model.h5'.format(os.sep)
        siamesemodel_path = '..{0}model{0}siamese_model.h5'.format(os.sep)

        self.configopts = {
            'imagedir' : imagedir,
            'augmenteddatadir': augmenteddatadir,
            'resnetmodel_path': resnetmodel_path,
            'siamesemodel_path': siamesemodel_path
        }
    
    def loadData(self):

        imagedir = self.configopts['imagedir']
        augmenteddatadir = self.configopts['augmenteddatadir']
        resnetmodel_path = self.configopts['resnetmodel_path']
        
        # generate augmented data
        #imageAugmentation = ImageAugmentation()
        #imageAugmentation.generateAugmentedData(imagedir, augmenteddatadir)

        # data preprocessing and feature extraction
        dataPreprocessing = DataPreprocessing(augmenteddatadir, resnetmodel_path, False)
        image_dataframe = dataPreprocessing.getDataFrameFromImage()
        siamese_dataframe = dataPreprocessing.extractImageFeaturesToSiameseDataFrame(image_dataframe, resnetmodel_path)

        train_df, val_df = train_test_split(siamese_dataframe, test_size=0.2)
        trfeatures_one, trfeatures_two, trlabels = dataPreprocessing.getSiameseDataset(train_df)
        valfeatures_one, valfeatures_two, vallabels = dataPreprocessing.getSiameseDataset(val_df)
        return trfeatures_one, trfeatures_two, trlabels, valfeatures_one, valfeatures_two, vallabels


    def trainModel(self):

        trfeatures_one, trfeatures_two, trlabels, valfeatures_one, valfeatures_two, vallabels = self.loadData()

        resnetmodel_path = self.configopts['resnetmodel_path']
        siamesemodel_path = self.configopts['siamesemodel_path']
        
        tr_onehotlabels = to_categorical(trlabels)
        val_onehotlabels = to_categorical(vallabels)
        
        fingerprintModel = FingerprintModel(fingerprint_modelpath=siamesemodel_path, featureextractor_modelpath=resnetmodel_path)
        print('training siamese network')
        fingerprintModel.train(modeltype='Siamese', 
                                predictor=[trfeatures_one, trfeatures_two],
                                response=tr_onehotlabels,
                                val_predictor=[valfeatures_one, valfeatures_two],
                                val_response=val_onehotlabels)

        # evaluate model on validation set
        fingerprintModel.evaluate(valfeatures_one, valfeatures_two, vallabels)
        
if __name__=='__main__':
     imagedir = '..{0}data{0}images'.format(os.sep)
    modelBuilder = ModelBuilder(imagedir)
    modelBuilder.trainModel()
