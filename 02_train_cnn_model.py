#!/usr/bin/python3

#import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from config import dataset_dir, image_size, objects_types, num_classes, epochs, batch_size


def create_simple_cnn(input_shape, num_classes=len(objects_types)):


    # Create a Sequential model
    model = models.Sequential()

    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output to feed into the fully connected layers
    model.add(layers.Flatten())

    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))

    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))  # For multi-class classification

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model):


    # Prepare the data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Set up directories for training and validation data
    train_dir = dataset_dir + '/images/train/'
    validation_dir = dataset_dir + '/images/validation/' 

    class_mode = 'binary' if len(objects_types)==2 else 'sparse'

    
    # Create the generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode=class_mode
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    return model, history








if __name__ == "__main__":

    # Example usage:
    model = create_simple_cnn(input_shape=(image_size, image_size, 3), num_classes=num_classes)
    model.summary(line_length=80)

    model, history = train_model(model)

    
    # Save the trained model
    from pathlib import Path
    path = Path('./model/')
    path.mkdir(parents=True, exist_ok=True)
    model.save('./model/trained_cnn_model.h5')























