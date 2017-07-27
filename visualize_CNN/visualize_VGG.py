from keras import backend as K 
from keras import applications

import numpy as np 

model = applications.VGG16(include_top = False,
							weights = 'imagenet')

layer_dict = dict([(layer.name, layer) for layer in model.layers])

for layer in model.layers:
	print(layer.name)
#
#
layer_name = 'block5_conv3'
#
filter_index = 0
#
layer_output = layer_dict[layer_name].output
#
loss = K.mean(layer_output[:,:,:,filter_index])
print(loss.shape)
#
#grads = K.gradients(loss, input_img)[0]
#
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
#iterate = k.function([input_img], [loss, grads])
#
#input_img_data
