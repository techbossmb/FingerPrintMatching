# Fingerprint Recognition with Siamese Network
Given a fingerprint image, model identifies a fingerprint match from a set of templates (Fingerprint DB) with the highest likelihood.

## Procedure
+ Training Fingerprint  Model<p>
    Given a set of fingerprint images, train a siamese network to identify similar <p>
    Fingerprints used during training are stored for future matching. To reduce computational time, we store only the extracted (intermediate) features. <p>
    Model is trained, validated and stored. <p>
    Model training code is in <code> code/train.py </code>.  <p>
    To run model training <p>
    ```python
    from train import ModelBuilder

    imagedir = '..{0}data{0}images'.format(os.sep)
    modelBuilder = ModelBuilder(imagedir)
    modelBilder.trainData()
    ```
    
    
+ Fingerprint Match
    After model has been trained, fingerprint matching modelfile is saved in <code>model/siamese_model.h5</code><p>
    To run a fingerprint match against a set of templates (stored during training)<p>
    ```python
    from match import FingerprintAuthentication
    
    matchthreshold = 0.8
    fingerprint_to_match = '..{0}data{0}images{0}101_4.tif'.format(os.sep)
    fingerprintAuthentication = FingerprintAuthentication()
    matched_tiffile, matched_prob, matched_personid = fingerprintAuthentication.matchFingerprint(fingerprint_to_match)
    print('Matched TIF: {}, Probability: {}, PersonID: {}'.format(matched_tiffile, matched_prob, matched_personid ))
    if matched_prob >= matchthreshold:
        print('Got a fingerprint match with {} for person id {}'.format(matched_tiffile, matched_personid))
    else:
        print('No match found')
    ```
    
## Model Architecture
Model is a siamese neural network with pre-trained ResNet50 as feature extractor<p>
<p>
![alt text](https://git.toptal.com/screening/isaac-ishola/raw/master/readme/model_architecture.JPG)

## Training and Validation Loss
Tensorboard  - Training and Validation cross entropy loss<p>
<p>
![alt text](https://git.toptal.com/screening/isaac-ishola/raw/master/readme/binarycrossentropy_graph.PNG)

## Sample Command Output
After the training completes, the validation metrics is outputed to stdout <p>
![alt text](https://git.toptal.com/screening/isaac-ishola/raw/master/readme/training_result.PNG)<p>
Result of fingerprint match<p>
![alt text](https://git.toptal.com/screening/isaac-ishola/raw/master/readme/fingerprintmatch_result.PNG)