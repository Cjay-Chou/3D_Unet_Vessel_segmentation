import sys
import os
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from tensorflow.python.keras import layers as klayers

from tensorflow import name_scope
import argparse
import re
from pathlib import Path
import random
import yaml

args = None

def ParseArgs():
    parser = argparse.ArgumentParser(description='This is a build 3D_U_Net program')
    parser.add_argument("datafile", help="Input Dataset file(stracture:data_path label_path)")
    parser.add_argument("-o", "--outfile", help="Output model structure file in YAML format (*.yml).")
    parser.add_argument("-t","--testfile", help="Input Dataset file for validation (stracture:data_path label_path)")
    parser.add_argument("-p", "--patchsize", help="Patch size. (ex. 44x44x28)", default="44x44x28")
    parser.add_argument("-c", "--nclasses", help="Number of classes of segmentaiton including background.", default=14, type=int)
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=30, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size*(Warning:memory use a lot)", default=3, type=int)
    parser.add_argument("-l", "--learningrate", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("--nobn", help="Do not use batch normalization layer", dest="use_bn", action='store_false')
    parser.add_argument("--nodropout", help="Do not use dropout layer", dest="use_dropout", action='store_false')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args


def CreateConv3DBlock(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
    for i in range(n):
        x = klayers.Conv3D(filters[i], (3,3,3), padding='valid', name=name+'_conv'+str(i+1))(x)
        if use_bn:
            x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
        x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)

    convresult = x

    if apply_pooling:
        x = klayers.MaxPool3D(pool_size=(2,2,2), name=name+'_pooling')(x)

    return x, convresult

def CreateUpConv3DBlock(x, contractpart, filters, n = 2, use_bn = True, name = 'upconvblock'):
    # upconv x
    x = klayers.Conv3DTranspose((int)(x.shape[-1]), (2,2,2), strides=(2,2,2), padding='same', use_bias = False, name=name+'_upconv')(x)
    # concatenate contractpart and x
    c = [(i-j)//2 for (i, j) in zip(contractpart[0].shape[1:4].as_list(), x.shape[1:4].as_list())]
    contract_crop = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[0])
    if len(contractpart) > 1:
        crop1 = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[1])
        #crop2 = klayers.Cropping3D(cropping=((c[0],c[0]),(c[1],c[1]),(c[2],c[2])))(contractpart[2])
        #x = klayers.concatenate([contract_crop, crop1, crop2, x])
        x = klayers.concatenate([contract_crop, crop1, x])
    else:
        x = klayers.concatenate([contract_crop, x])

    # conv x 2 times
    for i in range(n):
        x = klayers.Conv3D(filters[i], (3,3,3), padding='valid', name=name+'_conv'+str(i+1))(x)
        if use_bn:
            x = klayers.BatchNormalization(name=name+'_BN'+str(i+1))(x)
        x = klayers.Activation('relu', name=name+'_relu'+str(i+1))(x)

    return x

def Construct3DUnetModel(input_images, nclasses, use_bn = True, use_dropout = True):
    with name_scope("contract1"):
        x, contract1 = CreateConv3DBlock(input_images, (32, 64), n = 2, use_bn = use_bn, name = 'contract1')

    with name_scope("contract2"):
        x, contract2 = CreateConv3DBlock(x, (64, 128), n = 2, use_bn = use_bn, name = 'contract2')

    with name_scope("contract3"):
        x, contract3 = CreateConv3DBlock(x, (128, 256), n = 2, use_bn = use_bn, name = 'contract3')

    with name_scope("contract4"):
        x, _ = CreateConv3DBlock(x, (256, 512), n = 2, use_bn = use_bn, apply_pooling = False, name = 'contract4')

    with name_scope("dropout"):
        if use_dropout:
            x = klayers.Dropout(0.5, name='dropout')(x)

    with name_scope("expand3"):
        x = CreateUpConv3DBlock(x, [contract3], (256, 256), n = 2, use_bn = use_bn, name = 'expand3')

    with name_scope("expand2"):
        x = CreateUpConv3DBlock(x, [contract2], (128, 128), n = 2, use_bn = use_bn, name = 'expand2')

    with name_scope("expand1"):
        x = CreateUpConv3DBlock(x, [contract1], (64, 64), n = 2, use_bn = use_bn, name = 'expand1')

    with name_scope("segmentation"):
        layername = 'segmentation_{}classes'.format(nclasses)
        x = klayers.Conv3D(nclasses, (1,1,1), activation='softmax', padding='same', name=layername)(x)

    return x

def ReadSliceDataList(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            imagefile, labelfile = line.strip().split()
            datalist.append((imagefile, labelfile))

    return datalist

def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def weighted_crossentropy_loss(y_true, y_pred, axis = -1):
    K = tf.keras.backend
    y_pred /= tf.reduce_sum(y_pred, axis, True)
    # manual computation of crossentropy
    _epsilon = _to_tensor(1e-7, tf.float32)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    weight_y_org = tf.reduce_sum(y_true,[0,1,2,3])

    weight_y_3 = tf.pow(weight_y_org,1.0/3)
    weight_y = weight_y_3 / tf.reduce_sum(weight_y_3)

    return - tf.reduce_sum( 1 / (weight_y + _epsilon) * y_true * tf.log(y_pred), axis)

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def ImportImage(filename):
    image = sitk.ReadImage(filename)
    imagearry = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        imagearry = imagearry[..., np.newaxis]
    return imagearry

def GenerateBatchData(datalist, paddingsize, batch_size = 32):
    ps = paddingsize[::-1] # (x, y, z) -> (z, y, x) for np.array
    #j = 0

    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            imagelist = []
            outputlist = []

            for idx in indices[i:i+batch_size]:
                image = ImportImage(datalist[idx][0])
                onehotlabel = ImportImage(datalist[idx][1])

                onehotlabel = onehotlabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
                imagelist.append(image)
                outputlist.append(onehotlabel)

            yield (np.array(imagelist), np.array(outputlist))


def main(_):
    #Build 3DU-net
    matchobj = re.match("([0-9]+)x([0-9]+)x([0-9]+)", args.patchsize)
    if matchobj is None:
        print('[ERROR] Invalid patch size : {}'.format(args.patchsize))
        return
    patchsize = [ int(s) for s in matchobj.groups() ]
    patchsize = tuple(patchsize)

    padding = 44
    imagesize = tuple([ p + 2*padding for p in patchsize ]) 
    inputshape = imagesize[::-1] + (1,)
    nclasses = args.nclasses
    print("Input shape:", inputshape)
    print("Number of classes:", nclasses)

    inputs = tf.keras.layers.Input(shape=inputshape, name="input")
    segmentation = Construct3DUnetModel(inputs, nclasses, args.use_bn, args.use_dropout)

    model = tf.keras.models.Model(inputs, segmentation,name="3DUnet")
    model.summary()

    #Start training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)
        model.compile(loss=weighted_crossentropy_loss, optimizer=optimizer, metrics=[dice])
    
    if args.outfile is not None:
        with open(args.outfile, 'w') as f:
            yamlobj = yaml.load(model.to_yaml())
            yaml.dump(yamlobj, f)
            
    #get padding size
    ps = np.array(model.output_shape[1:4])[::-1]
    ips = np.array(model.input_shape[1:4])[::-1]
    paddingsize = ((ips - ps) / 2).astype(np.int)

    #A retraining of interruption
    if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile, by_name=True)
        initial_epoch = args.initialepoch


    if not os.path.exists(args.logdir+'/model'):
        os.makedirs(args.logdir+'/model')
    latestfile = args.logdir + '/latestweights.hdf5'
    bestfile = args.logdir + '/bestweights.hdf5'
    tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    best_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=bestfile, save_best_only = True)#, save_weights_only = True)
    latest_cbk = tf.keras.callbacks.ModelCheckpoint(filepath=latestfile)#, save_weights_only = True)
    every_cbk = tf.keras.callbacks.ModelCheckpoint(filepath = args.logdir + '/model/model_{epoch:02d}_{val_loss:.2f}.hdf5')
    callbacks = [tb_cbk,best_cbk,latest_cbk,every_cbk]

    #read dataset
    trainingdatalist = ReadSliceDataList(args.datafile)
    train_data = GenerateBatchData(trainingdatalist, paddingsize, batch_size = args.batchsize)
    if args.testfile is not None:
        testdatalist = ReadSliceDataList(args.testfile)
        #testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.3))
        validation_data = GenerateBatchData(testdatalist, paddingsize, batch_size = args.batchsize)
        validation_steps = len(testdatalist) / args.batchsize
    else:
        validation_data = None
        validation_steps = None

    steps_per_epoch = len(trainingdatalist) / args.batchsize
    print ("Number of samples:", len(trainingdatalist))
    print ("Batch size:", args.batchsize)
    print ("Number of Epochs:", args.epochs)
    print ("Learning rate:", args.learningrate)
    print ("Number of Steps/epoch:", steps_per_epoch)

    #with tf.device('/device:GPU:{}'.format(args.gpuid)):
    model.fit_generator(train_data,
            steps_per_epoch = steps_per_epoch,
            epochs = args.epochs,
            callbacks=callbacks,
            validation_data = validation_data,
            validation_steps = validation_steps,
            initial_epoch = initial_epoch )

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])