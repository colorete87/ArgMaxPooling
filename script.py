#!/usr/bin/python3


#import tensorflow as tf




seed = 123
dataset_dir = 'dataset'
image_size = 128
train_size = 1000
validation_size = 100
test_size = 100


objects_types = [
        'circle',
        'square',
        'cross',
        ]
num_classes = len(objects_types)



def create_dirs(output_dir):
    from pathlib import Path

    # Define the root directory
    root_dir = Path(output_dir)

    # Define the subdirectories to be created
    subdirs = [
        'images/train',
        'images/validation',
        'images/test',
        'argmax/train',
        'argmax/validation',
        'argmax/test'
    ]

    # Create each directory
    for subdir in subdirs:
        for type_ in objects_types:
            # Construct the full path
            path = root_dir / subdir / type_
            # Create the directory, including any necessary parent directories
            path.mkdir(parents=True, exist_ok=True)

    path = Path('./models/')
    path.mkdir(parents=True, exist_ok=True)

    print(f"Folder structure created under '{root_dir}'")



# Function to create a blank image and draw random objects
def create_random_object_image(image_size=(32, 32), seed=None):

    import random
    import numpy as np

    random.seed(seed)

    min_size = np.min(image_size)//20
    max_size = np.min(image_size)//2

    # Randomly select an object to draw: cross, circle, or square
    object_type = random.choice(['cross', 'circle', 'square'])

    # Randomize position, size, and thickness
    size = random.randint(min_size, max_size)
    center_x = random.randint(size//2+1, image_size[1] - size//2-1)
    center_y = random.randint(size//2+1, image_size[0] - size//2-1)
    thickness = random.randint(1, size//5)

    image = create_object_image(image_size, (center_x, center_y), size, thickness, object_type)

    return image, object_type, size, (center_x, center_y), thickness




def create_object_image(image_size, object_center, object_size, object_thickness, object_type):

    import cv2
    import numpy as np

    img_size = list(image_size) + [3]

    thickness = object_thickness
    size = object_size
    center_x = object_center[0]
    center_y = object_center[1]


    # Create a blank black image
    image = np.zeros(img_size, dtype=np.uint8)

    color = (255, 255, 255)  # White color for the objects

    if object_type == 'cross':
        # Draw a cross
        cv2.line(image, (center_x - size, center_y), (center_x + size, center_y), color, thickness)
        cv2.line(image, (center_x, center_y - size), (center_x, center_y + size), color, thickness)

    elif object_type == 'circle':
        # Draw a circle
        cv2.circle(image, (center_x, center_y), size, color, thickness)

    elif object_type == 'square':
        # Calculate the top-left and bottom-right points for the square
        top_left = (center_x - size, center_y - size)
        bottom_right = (center_x + size, center_y + size)
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    # Return the generated image
    return image





# Función para generar n imágenes
def generate_images(output_dir, image_size, train_size, validation_size, test_size, seed=None):

    from pathlib import Path
    import cv2
    import pickle

    # Generate directory structure
    create_dirs(output_dir)

    # Definir los tamaños y carpetas de salida
    sizes = {'train': train_size, 'validation': validation_size, 'test': test_size}
    infos = list()

    # Crear carpetas si no existen
    for subset, size in sizes.items():
        subset_dir = Path(output_dir + '/images') / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        # Generar y guardar las imágenes
        for i in range(size):
            image_params = create_random_object_image(image_size, seed)
            image = image_params[0]
            keys = ['type', 'size', 'center', 'thickness']
            params = image_params[1:]
            info = dict(zip(keys, params))
            info['set'] = subset
            info['index'] = i
            info['params'] = params
            filename = subset_dir / f"{params[0]}/{subset}_{i:010d}.png"
            cv2.imwrite(str(filename), image)
            infos.append(info)

    # Save the variable to a file
    with open(output_dir + '/dataset_info.pkl', 'wb') as file:  # 'wb' means write in binary mode
        pickle.dump(infos, file)

    print(f"Imágenes generadas y guardadas en '{output_dir}'")

    return infos



import tensorflow as tf
from tensorflow.keras import layers, models

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

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        batch_size=128,
        class_mode=class_mode
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=128,
        class_mode=class_mode
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    return model, history




if __name__ == "__main__":

    ## Generate and display the image
    #random_image = create_random_object_image(image_size=(image_size,image_size))[0]
    #import cv2
    #cv2.imshow('Random Object', random_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Llamada a la función para generar las imágenes
    infos = generate_images(dataset_dir,
                            image_size=(image_size,image_size),
                            train_size=1000,
                            validation_size=100,
                            test_size=100,
                            seed=seed,)

    # Example usage:
    model = create_simple_cnn(input_shape=(image_size, image_size, 3), num_classes=num_classes)
    model.summary()

    model, history = train_model(model)

    
    # Save the trained model
    model.save('trained_cnn_model.h5')

















import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

# Function to save the x and y indices of the max value for a MaxPooling layer
def save_maxpool_indices(layer_output, layer_name, output_dir):
    # layer_output shape: (1, height, width, channels)
    height, width, channels = layer_output.shape[1], layer_output.shape[2], layer_output.shape[3]

    # Initialize x and y index images
    x_indices_image = np.zeros((height, width, channels), dtype=np.uint8)
    y_indices_image = np.zeros((height, width, channels), dtype=np.uint8)

    # Iterate over each channel
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                # Get the 2x2 pooling window
                pooling_window = layer_output[0, i, j, c]

                # Find the index of the maximum value within the window
                max_index = np.argmax(pooling_window)
                max_y, max_x = np.unravel_index(max_index, (2, 2))

                # Save the indices in the corresponding x and y images
                x_indices_image[i, j, c] = max_x
                y_indices_image[i, j, c] = max_y

    # Save the images
    x_image_path = os.path.join(output_dir, f"{layer_name}_x_indices.png")
    y_image_path = os.path.join(output_dir, f"{layer_name}_y_indices.png")
    cv2.imwrite(x_image_path, x_indices_image)
    cv2.imwrite(y_image_path, y_indices_image)
    print(f"Saved x and y indices images for {layer_name}")

# Function to process image through layers and save max pooling indices
def process_image_and_save_maxpool_indices(model, img_path, output_dir, target_size=(128, 128)):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Create a model that outputs the activations of each layer
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Pass the image through the model
    activations = activation_model.predict(img_array)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each layer and process if it's a MaxPooling layer
    for layer, activation in zip(model.layers, activations):
        if 'max_pooling' in layer.name:  # Check if the layer is a MaxPooling layer
            save_maxpool_indices(activation, layer.name, output_dir)

# Example usage
# Assume `model` is your pre-trained Keras model and `img_path` is the path to your image
img_path = './dataset/images/train/cross/train_0000000227.png'
output_dir = './aux/'
process_image_and_save_maxpool_indices(model, img_path, output_dir)






