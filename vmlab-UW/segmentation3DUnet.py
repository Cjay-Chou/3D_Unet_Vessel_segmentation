import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np
import argparse
import yaml
import SimpleITK as sitk
from pathlib import Path

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="Input image file")
    parser.add_argument("modelfile", help="3D U-net model file (*.yml).")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("outfile", help="The filename of the segmented label image")
    parser.add_argument("--mask", help="The filename of mask image")
    parser.add_argument("--paoutfile", help="The filename of the estimated probabilistic map file.")
    parser.add_argument("--stepscale", help="Step scale for patch tiling.", default=1.0, type=float)
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args


def createParentPath(filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def Padding(image, patchsize, imagepatchsize, mirroring = False):
    padfilter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    padfilter.SetPadLowerBound(patchsize.tolist())
    padfilter.SetPadUpperBound(imagepatchsize.tolist())
    padded_image = padfilter.Execute(image)
    return padded_image


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print("loading 3D U-net model", args.modelfile, end="...", flush=True)
        with open(args.modelfile) as f:
            yamlobj = yaml.load(f)
            model = tf.keras.models.model_from_yaml(yaml.dump(yamlobj))
            modelversion = yamlobj["unet_version"] if "unet_version" in yamlobj else "v1"
        print("loading weights", end="...", flush=True)
        model.load_weights(args.modelweightfile)
        if modelversion == "v3":
            model = tf.keras.models.Model(inputs=model.input, outputs=model.output[0])
        print("done")
    print("input_shape =", model.input_shape)
    print("output_shape =", model.output_shape)

    # get patchsize
    ps = np.array(model.output_shape[1:4])[::-1]
    ips = np.array(model.input_shape[1:4])[::-1]
    ds = ((ips - ps) / 2).astype(np.int)
    
    print("loading input image", args.imagefile, end="...", flush=True)
    image = sitk.ReadImage(args.imagefile)
    image_padded = Padding(image, ds, ips, mirroring = True)
    print("done")
    s = image_padded.GetSize()
    print("ps:{}, ips:{}, ds:{}, s:{}".format(ps, ips, ds, s))

    bb = None
    maskarry = None
    if args.mask is None:
        bb = (ds[0], ds[0]+image.GetSize()[0]-1, ds[1], ds[1]+image.GetSize()[1]-1, ds[2], ds[2]+image.GetSize()[2]-1)
    else:
        print("loading mask image", args.mask, end="...", flush=True)
        maskimage = sitk.ReadImage(args.mask)
        if maskimage.GetPixelID() != sitk.sitkUInt8:
            maskimage = sitk.BinaryThreshold(maskimage, 1e-6)
        maskimage_padded = Padding(maskimage, ds, ips)
        statfilter=sitk.LabelStatisticsImageFilter()
        statfilter.Execute(image_padded, maskimage_padded)
        bb = statfilter.GetBoundingBox(1)
        maskarry = sitk.GetArrayFromImage(maskimage_padded)
        print("done")
    print("bb", bb)

    step = (ps / args.stepscale).astype(np.int8)

    totalpatches = [i for i in product( range(bb[4], bb[5], step[2]), range(bb[2], bb[3], step[1]), range(bb[0], bb[1], step[0]))]
    num_totalpatches = len(totalpatches)
    #patchindices = [i for i in product( range(bb[4], bb[5], ps[2]), range(bb[2], bb[3], ps[1]), range(bb[0], bb[1], ps[0]))]

    label = sitk.Image(image_padded.GetSize(), sitk.sitkUInt8)
    labelarr = sitk.GetArrayFromImage(label)
    #print('labelarr shape: {}'.format(labelarr.shape))
    counterarr = sitk.GetArrayFromImage(sitk.Image(image_padded.GetSize(), sitk.sitkVectorUInt8, model.output_shape[-1]))
    paarry = np.zeros(shape=(image_padded.GetSize()[::-1] + (model.output_shape[-1],)), dtype="float32")

    i = 1
    for iz in range(bb[4], bb[5], step[2]):
        for iy in range(bb[2], bb[3], step[1]):
            for ix in range(bb[0], bb[1], step[0]):
                p = [ix, iy, iz]
                print("patch [{} / {}] : {}".format(i, num_totalpatches, p), end="...", flush=True)
                i = i + 1
                ii = [p[0]-ds[0], p[1]-ds[1], p[2]-ds[2]]
                if ii[0]+ips[0] > s[0] or ii[1]+ips[1] > s[1] or ii[2]+ips[2] > s[2]:
                    print("skipped")
                    continue
                if maskarry is not None:
                    if np.sum(maskarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])]) < 1:
                        print("skipped")
                        continue

                patchimage = image_padded[ii[0]:(ii[0]+ips[0]), ii[1]:(ii[1]+ips[1]), ii[2]:(ii[2]+ips[2])]
                patchimagearray = sitk.GetArrayFromImage(patchimage)
                patchimagearray = np.array([patchimagearray[..., np.newaxis]])

                pavec = model.predict_on_batch(patchimagearray)
                #segmentation = np.argmax(pavec, axis=-1).astype(np.uint8)
                #labelarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])] = segmentation
                paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += pavec[0]
                counterarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += np.ones(pavec[0].shape, dtype = np.uint8)

                #if paarry is not None:
                #    paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] = pavec

                print("done")

    counterarr[counterarr == 0] = 1
    paarry = paarry / counterarr
    labelarr = np.argmax(paarry, axis=-1).astype(np.uint8)

    print("saving segmented label to", args.outfile, end="...", flush=True)
    createParentPath(args.outfile)
    labelarr = labelarr[ds[2]:-ips[2], ds[1]:-ips[1], ds[0]:-ips[0]]
    label = sitk.GetImageFromArray(labelarr)
    label.SetOrigin(image.GetOrigin())
    label.SetSpacing(image.GetSpacing())
    label.SetDirection(image.GetDirection())
    sitk.WriteImage(label, args.outfile, True)
    print("done")

    if args.paoutfile is not None:
        createParentPath(args.paoutfile)
        print("saving PA to", args.paoutfile, end="...", flush=True)
        paarry = paarry[ds[2]:-ips[2], ds[1]:-ips[1], ds[0]:-ips[0], :]
        pa = sitk.GetImageFromArray(paarry, isVector=True)
        pa.SetOrigin(image.GetOrigin())
        pa.SetSpacing(image.GetSpacing())
        pa.SetDirection(image.GetDirection())
        sitk.WriteImage(pa, args.paoutfile, True)
        print("done")

    tf.keras.backend.clear_session()

if __name__ == '__main__':
    args = ParseArgs()
    tf.app.run(main=main, argv=[sys.argv[0]])
