

import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, LeakyReLU, MaxPool2D, BatchNormalization, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam 




def create_model(trial):
    
    # Hyperparameters to be tuned by Optuna.
    
    #momentum = trial.suggest_float("momentum", 0.0, 1.0)
    filters = trial.suggest_categorical('filters', [32, 64])  #for filters in Conv2D layer 
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5]) # for kernel size in Conv2D layer
    strides = trial.suggest_categorical('strides', [1, 2]) # for stride in Conv2D layer
    activation = trial.suggest_categorical('activation', ['relu', 'linear'])
    units = trial.suggest_categorical("units", [32, 64, 128, 256, 512]) # for units in Dense layer  
    rate = trial.suggest_float('rate', 0.2, 0.6) # for rate in Dropout layer 

    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    beta_1 = trial.suggest_float('beta_1', 0.9, 0.999, log=True)
    beta_2 = trial.suggest_float('beta_2', 0.9, 0.9999, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-08, 1e-07)
    

    ##############################################
    # First Model (Pretrained Model)
    ##############################################
    base_model = DenseNet201(input_shape=(256, 256, 3), include_top=False, weights="imagenet")
    x = base_model.output
    x = BatchNormalization()(x)
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    x = Dense(350, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model1 = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False


    ###################################################
    # Own Built Model(Second Model)
    ##################################################
    
    model2 = Sequential()
    model2.add(
        Conv2D(
            filters=trial.suggest_categorical('filters', [32, 64]),
            kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
            strides = trial.suggest_categorical('strides', [1, 2]),
            activation=trial.suggest_categorical('activation', ['relu', 'linear']),
            input_shape=(256, 256, 3) 
        )
    )
    model2.add(BatchNormalization(momentum=0.5, epsilon=1e-5, gamma_initializer="uniform"))
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(Conv2D(64, (3,3), padding='same' ))
    model2.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(MaxPooling2D(2,2))
    model2.add(Dropout(0.2)) 

    model2.add(Conv2D(128, (3,3), padding='same' ))
    model2.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(Conv2D(128, (3,3), padding='same' ))
    model2.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(MaxPooling2D(2,2))
    model2.add(Dropout(0.2)) 


    model2.add(Conv2D(256, (3,3), padding='same' ))
    model2.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform" ))
    model2.add(LeakyReLU(alpha=0.1 ))
    model2.add(Conv2D(256, (3,3), padding='same' ))
    model2.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform" ))
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(MaxPooling2D(2,2) ) 
    model2.add(
        Dropout(
            rate=rate
        ) 
    ) 

    model2.add(Flatten())
    model2.add(
        Dense(
            units=units
        ) 
    )
    model2.add(LeakyReLU(alpha=0.1) )
    model2.add(BatchNormalization() )
    model2.add(Dense(3, activation='softmax') )
        

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    # Compile model.
    model2.compile(
        optimizer= optimizer_name,            #Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model2
