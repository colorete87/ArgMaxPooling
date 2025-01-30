
#!/usr/bin/python3


#import tensorflow as tf




seed = 123

dataset_dir = 'dataset'

image_size = 128

train_size = 1000
validation_size = 100
test_size = 100
epochs = 20
batch_size = 32


objects_types = [
        'circle',
        'square',
        'cross',
        'rectangle',
        ]
num_classes = len(objects_types)

