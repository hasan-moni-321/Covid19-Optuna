import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_dataset():
    train_datagen = ImageDataGenerator(  
                          shear_range=0.2, 
                          zoom_range=0.2,  
                          horizontal_flip=True,                
                          rescale=1./255, 
                          )
    
    train_data_generator = train_datagen.flow_from_directory('/home/hasan/Data Set/covid19/COVID-19 Radiography Database/train',
                                                  target_size=(256,256),
                                                  batch_size=32,
                                                  class_mode='categorical')
    return train_data_generator


def eval_dataset():
    test_datagen = ImageDataGenerator(
                            rescale=1./255
                            )

    test_data_generator = test_datagen.flow_from_directory('/home/hasan/Data Set/covid19/COVID-19 Radiography Database/test',
                                                       target_size=(256,256),
                                                       batch_size=32,
                                                       class_mode='categorical'
                                                       )
    return test_data_generator

