
#!/usr/bin/python3


#import tensorflow as tf




seed = 123

dataset_dir = 'dataset'

image_size = 32

train_size = 100000
validation_size = 10000
test_size = 10000
epochs = 10
batch_size = 256


objects_types = [
        'circle',
        'square',
        #'cross',
        #'rectangle',
        ]
num_classes = len(objects_types)

