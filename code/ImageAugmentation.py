import os
from glob import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class ImageAugmentation:
    
    def __init__(self):
        self.datagenerator = ImageDataGenerator(
                    rotation_range=15,
                    horizontal_flip=True,
                    vertical_flip=True,
                    zoom_range=(0.9,1.1),
                    brightness_range=(0.7,1.3),
                    fill_mode='constant',
                    cval=220)

    
    def generateAugmentedData(self, datadir, outputdir):
        print('Generating Augmented Data')
        imagefiles = glob('{}{}*.tif'.format(datadir, os.sep))
        for imagefile in imagefiles:
            fileid = imagefile.split(os.sep)[-1].split('.')[0]
            img = load_img(imagefile)
            imagedata = img_to_array(img)
            imagedata = imagedata.reshape((1,) + imagedata.shape) 

            i = 0
            for batch in self.datagenerator.flow(imagedata, batch_size=1,
                                    save_to_dir=outputdir, save_prefix='{}aug'.format(fileid), save_format='tif'):
                i += 1
                if i == 10: # generated 10 additional samples
                    break
