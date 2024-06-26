import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

def load_data(csv_file, img_size=128):
    data = pd.read_csv(csv_file)
    images, emotions, animal_types = [], [], []

    for _, row in data.iterrows():
        image_path = row['animal_images']
        emotion = row['animal_emotions']
        animal_type = row['animal_type']
        
        image = load_img(image_path, target_size=(img_size, img_size))
        image = img_to_array(image) / 255.0
        
        images.append(image)
        emotions.append(emotion)
        animal_types.append(animal_type)

    return np.array(images), np.array(emotions), np.array(animal_types)

def create_model(img_size=128, num_emotions=3, num_animals=10):
    inputs = Input(shape=(img_size, img_size, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    
    emotion_output = Dense(num_emotions, activation='softmax', name='emotion')(x)
    animal_output = Dense(num_animals, activation='softmax', name='animal')(x)
    
    model = Model(inputs=inputs, outputs=[emotion_output, animal_output])
    model.compile(optimizer='adam',
                 loss={'emotion': 'sparse_categorical_crossentropy', 'animal': 'sparse_categorical_crossentropy'},
                 metrics={'emotion': 'accuracy', 'animal': 'accuracy'})
    return model

def train_model(csv_file, img_size=128, epochs=100):
    images, emotions, animal_types = load_data(csv_file, img_size)
    
    emotion_encoder = LabelEncoder()
    animal_encoder = LabelEncoder()
    
    emotions = emotion_encoder.fit_transform(emotions)
    animal_types = animal_encoder.fit_transform(animal_types)

    X_train, X_test, y_train_emotions, y_test_emotions, y_train_animals, y_test_animals = train_test_split(
        images, emotions, animal_types, test_size=0.2, random_state=42
    )
    
    model = create_model(img_size, len(emotion_encoder.classes_), len(animal_encoder.classes_))
    model.fit(X_train, {'emotion': y_train_emotions, 'animal': y_train_animals}, 
             epochs=epochs, validation_data=(X_test, {'emotion': y_test_emotions, 'animal': y_test_animals}))

    logger.info("Model trained successfully")
    return model, emotion_encoder, animal_encoder

def predict(model, image_path, img_size=128, emotion_encoder=None, animal_encoder=None):
    if emotion_encoder is None or animal_encoder is None:
        raise ValueError("Both emotion_encoder and animal_encoder must be provided")
    
    image = load_img(image_path, target_size=(img_size, img_size))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    emotion_pred, animal_pred = model.predict(image)
    emotion = emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
    animal = animal_encoder.inverse_transform([np.argmax(animal_pred)])[0]
    
    return emotion, animal