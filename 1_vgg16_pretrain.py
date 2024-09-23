from network import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
import argparse
from time import time


 
img_size=224
ap = argparse.ArgumentParser()
ap.add_argument("-train","--train_dir",type=str, required=True,help="(required) the train data directory")
ap.add_argument("-val","--val_dir",type=str, required=True,help="(required) the validation data directory")
ap.add_argument("-num_class","--class",type=int, default=2,help="(required) number of classes to be trained")

args = vars(ap.parse_args())



batch_size=10

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
        args["train_dir"],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        args["val_dir"],
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')


print('loading the model and the pre-trained weights...')

base_model = VGG16.VGG16(include_top=False, weights='imagenet')
## Here we will print the layers in the network
i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
    print(i,layer.name)
#sys.exit()

x = base_model.output
x = Dense(128)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(args["class"], activation='softmax')(x)




tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

filepath = 'cv-tricks_pretrained_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
callbacks_list = [checkpoint,tensorboard]


model = Model(inputs=base_model.input, outputs=predictions)

#model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),metrics=["accuracy"])

num_training_img=1000
num_validation_img=400
stepsPerEpoch = num_training_img/batch_size
validationSteps= num_validation_img/batch_size
model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=20,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps
        )


