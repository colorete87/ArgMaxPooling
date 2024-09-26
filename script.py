#!/usr/bin/python3


#import tensorflow as tf




dataset_dir = 'dataset'
image_size = 128
train_size = 1000
validation_size = 100
test_size = 100





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
        # Construct the full path
        path = root_dir / subdir
        # Create the directory, including any necessary parent directories
        path.mkdir(parents=True, exist_ok=True)

    print(f"Folder structure created under '{root_dir}'")



# Function to create a blank image and draw random objects
def create_random_object_image(image_size=(32, 32)):

    import random
    import numpy as np

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

    return image, size, (center_x, center_y), thickness




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
def generate_images(output_dir, image_size, train_size, validation_size, test_size):

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
            image_params = create_random_object_image(image_size)
            image = image_params[0]
            keys = ['size', 'center', 'thickness']
            params = image_params[1:]
            info = dict(zip(keys, params))
            info['set'] = subset
            info['index'] = i
            info['params'] = params
            filename = subset_dir / f"{subset}_{i}.png"
            cv2.imwrite(str(filename), image)
            infos.append(info)

    # Save the variable to a file
    with open(output_dir + '/images/dataset_info.pkl', 'wb') as file:  # 'wb' means write in binary mode
        pickle.dump(infos, file)

    print(f"Imágenes generadas y guardadas en '{output_dir}'")

    return infos







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
                            test_size=100)



