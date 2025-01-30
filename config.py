
#!/usr/bin/python3


#import tensorflow as tf




seed = 123

dataset_dir = 'dataset'

image_size = 128

train_size = 10000
validation_size = 1000
test_size = 1000
epochs = 20
batch_size = 128


objects_types = [
        'circle',
        'square',
        'cross',
        'rectangle',
        ]
num_classes = len(objects_types)

