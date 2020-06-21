import os
from DataPreprocessing import DataPreprocessing
from FingerprintModel import FingerprintModel

class FingerprintAuthentication:
    def __init__(self):
        datadir = '..{0}data{0}images'.format(os.sep)
        resnetmodel_path = '..{0}model{0}resnet_model.h5'.format(os.sep)
        siamesemodel_path = '..{0}model{0}siamese_model.h5'.format(os.sep)
        
        self.dataPreprocessing = DataPreprocessing(datadir, resnetmodel_path)
        self.fingerprintModel = FingerprintModel(siamesemodel_path, resnetmodel_path)
        self.keylist = list(self.dataPreprocessing.fingerprintDatabase.keys())

    def  matchFingerprint(self, fingerprint, person_to_match=None):

        # no personid is specified, match with whole DB 
        if person_to_match is None:
            siamesefeature_one, siamesefeature_two = self.dataPreprocessing.createTestData(fingerprint)
        else:
            template_DB_keylist = []
            siamesefeature_one, siamesefeature_two = self.dataPreprocessing.createTestDataWithPersonID(fingerprint, person_to_match, template_DB_keylist)
        match_index, matched_prob = self.fingerprintModel.predict(siamesefeature_one, siamesefeature_two)
        DBkeylist = self.keylist if person_to_match is None else template_DB_keylist
        person_id = DBkeylist[match_index].split('aug')[0].split('_')[0]
        matched_tif = DBkeylist[match_index]
        matched_person = self.dataPreprocessing.person_guid_map[person_id]
        return matched_tif, matched_prob, matched_person

if __name__=='__main__':
    fingerprint_to_match = '..{0}data{0}images{0}107_3.tif'.format(os.sep)
    person_to_match = 'ynvklcdshb'
    threshold = 0.8 # need to find optimum threshold value
    fingerprintAuthentication = FingerprintAuthentication()
    matched_tif, matched_prob, matched_person = fingerprintAuthentication.matchFingerprint(fingerprint_to_match, person_to_match)
    print('Matched TIF: {}, Probability: {}, PersonID: {}'.format(matched_tif, matched_prob, matched_person ))
    if matched_prob >= threshold:
        print('Got a fingerprint match with {} for person id {}'.format(matched_tif, matched_person))
    else:
        print('No match found')