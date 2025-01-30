#!/usr/bin/python3


























#import tensorflow as tf
#import numpy as np
#from tensorflow.keras.preprocessing import image
#import cv2
#import os
#
## Function to save the x and y indices of the max value for a MaxPooling layer
#def save_maxpool_indices(layer_output, layer_name, output_dir):
#    # layer_output shape: (1, height, width, channels)
#    height, width, channels = layer_output.shape[1], layer_output.shape[2], layer_output.shape[3]
#
#    # Initialize x and y index images
#    x_indices_image = np.zeros((height, width, channels), dtype=np.uint8)
#    y_indices_image = np.zeros((height, width, channels), dtype=np.uint8)
#
#    # Iterate over each channel
#    for c in range(channels):
#        for i in range(height):
#            for j in range(width):
#                # Get the 2x2 pooling window
#                pooling_window = layer_output[0, i, j, c]
#
#                # Find the index of the maximum value within the window
#                max_index = np.argmax(pooling_window)
#                max_y, max_x = np.unravel_index(max_index, (2, 2))
#
#                # Save the indices in the corresponding x and y images
#                x_indices_image[i, j, c] = max_x
#                y_indices_image[i, j, c] = max_y
#
#    # Save the images
#    x_image_path = os.path.join(output_dir, f"{layer_name}_x_indices.png")
#    y_image_path = os.path.join(output_dir, f"{layer_name}_y_indices.png")
#    cv2.imwrite(x_image_path, x_indices_image)
#    cv2.imwrite(y_image_path, y_indices_image)
#    print(f"Saved x and y indices images for {layer_name}")
#
## Function to process image through layers and save max pooling indices
#def process_image_and_save_maxpool_indices(model, img_path, output_dir, target_size=(128, 128)):
#    # Load and preprocess the image
#    img = image.load_img(img_path, target_size=target_size)
#    img_array = image.img_to_array(img) / 255.0  # Normalize
#    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
#    # Create a model that outputs the activations of each layer
#    layer_outputs = [layer.output for layer in model.layers]
#    activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
#    
#    # Pass the image through the model
#    activations = activation_model.predict(img_array)
#
#    # Create output directory if it doesn't exist
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    # Iterate over each layer and process if it's a MaxPooling layer
#    for layer, activation in zip(model.layers, activations):
#        if 'max_pooling' in layer.name:  # Check if the layer is a MaxPooling layer
#            save_maxpool_indices(activation, layer.name, output_dir)
#
## Example usage
## Assume `model` is your pre-trained Keras model and `img_path` is the path to your image
#img_path = './dataset/images/train/cross/train_0000000227.png'
#output_dir = './aux/'
#process_image_and_save_maxpool_indices(model, img_path, output_dir)






