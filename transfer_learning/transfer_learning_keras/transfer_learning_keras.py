from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.models import optimizers
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height = 256, 256
train_data_dir = "data/train"
validation_data_dir = "data/val"
nb_train_samples = 4125
nb_validation_samples = 466
batch_size = 16
epochs = 50

model = applications.VGG16(weights = 'imagenet', include_top = False,
        input_shape = (img_width, img_height, 3))

for layer in model.layers:
    layer.trainable = False

#model.summary()
x = model.output
x = Flatten()(x)
x = Dense(1024,activation = 'relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation = 'relu')(x)
predications = Dense(16, activation = 'softmax')(x)
model_final = Model(input = model.input, output = predications)
#model_final.summary()

model_final.compile(loss = 'categorical_crossentropy', 
        optimizer = optimizers.SGD(lr = 0.0001, momentum = 0.9), 
        metrics = ['accuracy'])

train_data_generator = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = 'nearest',
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        rotation_range = 30)
test_data_generator = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
        fill_mode = 'nearest',
        zoom_range = 0.3,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        rotation_range = 30)

train_generator = train_data_generator.flow_from_dictinary(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'categorical')

test_generator = test_data_generator.flow_from_dictinary(
        test_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'categorical')

checkpoint = ModelCheckpoint('vgg16.h5', monitor = 'val_acc', verbose = 1,
                              save_best_only = True, save_weights_only = False,
                              mode = 'auto', period = 1)
early = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 10, verbose = 1, mode = 'auto')

model_final.fit_generator(
        train_generator,
        sample_per_epoch = nb_train_samples,
        epochs = epochs,
        validation_data = validation_generator,
        nb_val_samples = nb_validation_samples,
        callbacks = [checkpint, early])

