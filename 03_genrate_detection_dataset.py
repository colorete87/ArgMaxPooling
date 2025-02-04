#!/usr/bin/python3

#from config import dataset_dir, image_size, objects_types, train_size, validation_size

import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress tracking
import tensorflow as tf

from config import dataset_dir


def load_dataset_info():
    import pickle

    pickle_file = dataset_dir + '/dataset_info.pkl'

    # Load the data
    with open(pickle_file, "rb") as file:  # 'rb' means read in binary mode
        loaded_infos = pickle.load(file)

    ## Display the first few records to verify the data
    #for info in loaded_infos[:5]:  # Print first 5 entries
    #    print(info)

    return loaded_infos




## Función para obtener los índices de los valores máximos
#def get_image_info(image, pool_size):
#    h, w, c = image.shape[0] // pool_size[0], image.shape[1] // pool_size[1], image.shape[2]
#    indices_x = np.zeros((h, w, c), dtype=np.int32)
#    indices_y = np.zeros((h, w, c), dtype=np.int32)
#    
#    for i in range(h):
#        for j in range(w):
#            for k in range(c):
#                window = image[i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1], k]
#                max_idx = np.unravel_index(np.argmax(window, axis=None), window.shape)
#                indices_x[i, j, k] = max_idx[1]  # Índice x dentro de la ventana
#                indices_y[i, j, k] = max_idx[0]  # Índice y dentro de la ventana
#                max_value[i, j, k] =
#                mean[i, j, k] =
#                std[i, j, k] =
#                
#    return indices_x, indices_y, max_value, mean, std

def get_image_info(image, pool_size):
    h, w, c = image.shape[0] // pool_size[0], image.shape[1] // pool_size[1], image.shape[2]
    
    indices_x = np.zeros((h, w, c), dtype=np.int32)
    indices_y = np.zeros((h, w, c), dtype=np.int32)
    max_value = np.zeros((h, w, c), dtype=np.float32)
    mean = np.zeros((h, w, c), dtype=np.float32)
    std = np.zeros((h, w, c), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            for k in range(c):
                # Extract the pooling window
                window = image[i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1], k]
                
                # Compute max value and its indices
                max_idx = np.unravel_index(np.argmax(window, axis=None), window.shape)
                indices_x[i, j, k] = max_idx[1]  # X index of max value
                indices_y[i, j, k] = max_idx[0]  # Y index of max value
                max_value[i, j, k] = window[max_idx]  # Store the max value
                
                # Compute mean and standard deviation
                mean[i, j, k] = np.mean(window)
                std[i, j, k] = np.std(window)
                
    return indices_x, indices_y, max_value, mean, std



# Load dataset metadata
infos = load_dataset_info()

#infos = infos[0:10]


## Read first image to get dimensions
#sample_image_path = Path(dataset_dir) / 'images' / infos[0]['set'] / f"{infos[0]['type']}/{infos[0]['set']}_{infos[0]['index']:010d}.png"
#sample_image = cv2.imread(str(sample_image_path), cv2.IMREAD_UNCHANGED)
#if sample_image is None:
#    raise ValueError(f"Sample image not found: {sample_image_path}")



if __name__ == "__main__":

    df = pd.DataFrame()  # Create an empty DataFrame to store the metadata

    # Define the model path
    model_path = Path("./models/trained_cnn_model.h5")

    # Check if the model file exists
    if model_path.exists():
        model = tf.keras.models.load_model(str(model_path))
        print("Model loaded successfully!")
    else:
        print(f"Error: Model file not found at {model_path}")

    for info in tqdm(infos, desc="Processing Images", unit="image"):
        subset = info['set']
        index = info['index']
        type_ = info['type']

        subset_dir = Path(dataset_dir) / 'images' / subset
        filename = subset_dir / f"{type_}/{subset}_{index:010d}.png"

        # Load the image
        image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)

        in_ = model.layers[0](tf.constant(image[None,...], dtype=tf.float32))

        #print(in_.numpy())

        for layer in model.layers[1:]:
            out = layer(in_)
            aux_image = in_.numpy().squeeze()
            in_ = out

            if 'max_pooling2d' not in layer.name:
                continue

            pool_size = layer.pool_size

            # Get image dimensions
            img_height, img_width, img_channels = aux_image.shape
            img_height, img_width, img_channels = img_height//pool_size[0], img_width//pool_size[1], img_channels
            num_pixels = img_height * img_width * img_channels

            # Create column names for each pixel
            columns_x = [f"x_{i}" for i in range(num_pixels)]
            columns_y = [f"y_{i}" for i in range(num_pixels)]


            # Flatten image pixel
            if aux_image is not None:
                indices_x, indices_y, max_value, mean, std = get_image_info(aux_image, pool_size)
                flattened_indices_x = indices_x.flatten() * 2 - 1
                flattened_indices_y = indices_y.flatten() * 2 - 1
            else:
                print(f"Warning: Image not found - {filename}")
                flattened_pixels = None  # Handle missing images gracefully

            # Create a DataFrame for the new row
            new_row = pd.DataFrame([{
                'set': subset,
                'index': index,
                'type': type_,
                'size_x': info['size_x'],
                'size_y': info['size_y'],
                'center_x': info['center_x'],
                'center_y': info['center_y'],
                'thickness': info['thickness'],
                #'pixels': flattened_pixels
                #**dict(zip(pixel_columns, flattened_pixels))  # Add pixel values as columns
                **dict(zip(columns_x, flattened_indices_x)),  # Add pixel values as columns
                **dict(zip(columns_y, flattened_indices_y)),  # Add pixel values as columns
            }])

            # Append using pd.concat()
            df = pd.concat([df, new_row], ignore_index=True)

    # Save DataFrame as CSV for future use
    df.to_csv(Path(dataset_dir) / "dataset_metadata.csv", index=False)

    print(f"Dataset metadata saved: {dataset_dir}/dataset_metadata.csv")

    # Display first few rows
    print(df.head())

