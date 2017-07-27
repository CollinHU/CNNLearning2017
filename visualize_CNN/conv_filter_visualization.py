from scipy.misc import imsave
import numpy as np 
import time
from keras.applications import vgg16
from keras import  backend as K 

img_width = 128
img_height = 128

layer_name = 'block5_conv1'

def deprocess_image(x):
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	x *= 255

	if K.image_data_format() == 'channels_first':
		x = x.transpose((1,2,0))
	x = np.clip(x, 0, 255).astype('uint8')

	return x

model = vgg16.VGG16(weights = 'imagenet', include_top = False)

print('Modle loaded')

model.summary()

#  this is placeholder for the input images
input_img = model.input

layer_dic = dict([(layer.name,layer) for layer in model.layers[1:]])


def normalize(x):
	return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

kept_filters = []

for filter_index in range(0, 50):
	print('Processing filter %d' % filter_index)

	start_time = time.time()

	layer_output = layer_dic[layer_name].output 

	if K.image_data_format() == 'channel_first':
		loss = K.mean(layer_output[:, filter_index, :, :])
	else:
		loss = K.mean(layer_output[:, :, :, filter_index])

	grads = K.gradients(loss, input_img)[0]

	grads = normalize(grads)

	iterate = K.function([input_img], [loss, grads])

	step = 1

	if K.image_data_format == 'channel_first':
		input_img_data = np.random.random((1,3,img_width,img_height))
	else:
		input_img_data = np.random.random((1,img_width,img_height,3))

	input_img_data = (input_img_data - 0.5) * 20 + 128
	for i in range(20):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

		print('current loss value: ', loss_value)

		if loss_value <= 0.:
			break
	if loss_value > 0:
		img = deprocess_image(input_img_data[0])
		kept_filters.append((img, loss_value))

	end_time = time.time()

	print('Filter %d processed in %d s '%(filter_index, end_time - start_time))

n = 2

kept_filters.sort(key = lambda x: x[1], reverse = True)

kept_filters = kept_filters[:n*n]

margin = 5

width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin

stitched_filters = np.zeros((width, height, 3))

for i in range(n):
	for j in range(n):
		img, loss = kept_filters[i * n + j]
		stitched_filters[(img_width + margin) * i : (img_width + margin) * i + img_width,
						 (img_height + margin) * j : (img_height + margin) * j + img_height, : ] = img 

imsave('stitched_filters_%dx%d.png' %(n ,n), stitched_filters)








