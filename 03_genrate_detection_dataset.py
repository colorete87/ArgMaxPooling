#!/usr/bin/python3

#from config import dataset_dir, image_size, objects_types, train_size, validation_size

import numpy as np

# Función para obtener los índices de los valores máximos
def get_max_indices(image, pool_size):
    h, w, c = image.shape[0] // pool_size[0], image.shape[1] // pool_size[1], image.shape[2]
    indices_x = np.zeros((h, w, c), dtype=np.int32)
    indices_y = np.zeros((h, w, c), dtype=np.int32)
    
    for i in range(h):
        for j in range(w):
            for k in range(c):
                window = image[i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1], k]
                max_idx = np.unravel_index(np.argmax(window, axis=None), window.shape)
                indices_x[i, j, k] = max_idx[1]  # Índice x dentro de la ventana
                indices_y[i, j, k] = max_idx[0]  # Índice y dentro de la ventana
                
    return indices_x, indices_y




