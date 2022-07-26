import os
import argparse
import re
import subprocess
import tensorflow as tf
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input-shape', type=int, default=224)
parser.add_argument('--tflite-model-folder', required=True, type=str)
parser.add_argument('--tpu-model-folder', required=True, type=str)
parser.add_argument('--image-path', required=True, type=str)
args, unknown_args = parser.parse_known_args()

input_shape = (args.input_shape, args.input_shape, 3)


def compile_to_edgetpu(tflite_model_file):
    command_arguments = ('/usr/bin/edgetpu_compiler', '-o', args.tpu_model_folder, '-m',
                         '13', '-s', os.path.join(args.tflite_model_folder, tflite_model_file))

    log_file_name = re.sub('\\.tflite$', '.log', os.path.basename(tflite_model_file))
    
    results = subprocess.run(command_arguments,
                             capture_output=True,
                             check=True)
    
    with open(os.path.join(args.tpu_model_folder, log_file_name), 'wt') as log_file:
        log_file.write(results.stderr.decode())
        log_file.write(results.stdout.decode())


def reset_range(input_array):
    result = input_array.astype(np.float32)
    result = result.reshape(tuple([1] + list(result.shape)))
    return result


class InputDataGenerator(tf.lite.RepresentativeDataset):
    def __init__(self, num_samples = 300):
        self.num_samples = num_samples
        super().__init__(self.sample_generator)

    def sample_generator(self):
        src = cv2.imread(args.image_path)
        image = cv2.resize(src, (args.input_shape, args.input_shape))
        for i in range(self.num_samples):
            yield [reset_range(image)]

            
def doit():
    
    model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    x = model.layers[-1].output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model_before_output = x
    new_output_layer = tf.keras.layers.Dense(128, activation=None)(model_before_output)
    output_layers = []
    model.layers[0]._name = 'image_input'
    output_layers.append(new_output_layer)
    network = tf.keras.Model(inputs=model.input,
                                      outputs=output_layers,
                                      name=f'densenet_new_net')
    converter = tf.lite.TFLiteConverter.from_keras_model(network)
    converter.experimental_new_converter = True  
    converter.optimizations = [tf.lite.Optimize.DEFAULT] 
     
    converter.representative_dataset = InputDataGenerator(num_samples = 300)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    
    tflite_model_file = 'Densenet121_' + str(args.input_shape) + '_int.tflite'
        
    with open(os.path.join(args.tflite_model_folder, tflite_model_file), 'wb') as tflite_out_file:
        tflite_out_file.write(tflite_model)
    
    compile_to_edgetpu(tflite_model_file)


doit()
