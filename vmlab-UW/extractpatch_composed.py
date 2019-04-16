import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np
import argparse
import yaml
import SimpleITK as sitk
from pathlib import Path
from keras.utils import to_categorical



args = None


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile", help="Input image file")
    parser.add_argument("modelfile", help="3D U-net model file (*.yml).")
    parser.add_argument("outpath", help="The cut path")
    parser.add_argument("listpath", help="The cut list")
    parser.add_argument("--mask", help="The filename of mask image")
    parser.add_argument("--paoutfile", help="The filename of the estimated probabilistic map file.")
    parser.add_argument("--stepscale", help="Step scale for patch tiling.", default=1.0, type=float)
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args

args = ParseArgs()
def Padding(image, patchsize, imagepatchsize, mirroring = False):
    padfilter = sitk.MirrorPadImageFilter() if mirroring else sitk.ConstantPadImageFilter()
    padfilter.SetPadLowerBound(patchsize.tolist())
    padfilter.SetPadUpperBound(imagepatchsize.tolist())
    padded_image = padfilter.Execute(image)
    return padded_image

def createParentPath(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        print("path created")


print("loading 3D U-net model", args.modelfile, end="...", flush=True)
with open(args.modelfile) as f:
    yamlobj = yaml.load(f)
    model = tf.keras.models.model_from_yaml(yaml.dump(yamlobj))
    modelversion = yamlobj["unet_version"] if "unet_version" in yamlobj else "v1"



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

step = ps.astype(np.int8)


bb = None
maskarry = None


bb = (ds[0], ds[0]+image.GetSize()[0]-1, ds[1], ds[1]+image.GetSize()[1]-1, ds[2], ds[2]+image.GetSize()[2]-1)
print("bb", bb)

if args.mask is not None:
    print("loading mask image", args.mask, end="...", flush=True)
    maskimage = sitk.ReadImage(args.mask)
    maskimage_padded = Padding(maskimage, ds, ips)
    maskarry = sitk.GetArrayFromImage(maskimage_padded)
    print("done")

totalpatches = [i for i in product( range(bb[4], bb[5], step[2]), range(bb[2], bb[3], step[1]), range(bb[0], bb[1], step[0]))]
num_totalpatches = len(totalpatches)

image_padded_array=sitk.GetArrayFromImage(image_padded)

array_categorical=to_categorical(image_padded_array)

i = 1
j = 1
arr_path = []
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


            #patchimage = image_padded[ii[0]:(ii[0]+ips[0]), ii[1]:(ii[1]+ips[1]), ii[2]:(ii[2]+ips[2])]

            patchimagearray = array_categorical[ii[2]:(ii[2]+ips[2]), ii[1]:(ii[1]+ips[1]), ii[0]:(ii[0]+ips[0]), :]
            print(patchimagearray.shape)
            #pavec = model.predict_on_batch(patchimagearray)
            #segmentation = np.argmax(pavec, axis=-1).astype(np.uint8)
            #labelarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0])] = segmentation
            #paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += pavec[0]
            #counterarr[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] += np.ones(pavec[0].shape, dtype = np.uint8)

            #if paarry is not None:
            #    paarry[p[2]:(p[2]+ps[2]), p[1]:(p[1]+ps[1]), p[0]:(p[0]+ps[0]), :] = pavec

            print("saving cut to", args.outpath, end="...", flush=True)
            createParentPath(args.outpath)
            outfile = os.path.join(args.outpath,"image{}.mha".format(j))
            #labelarr = labelarr[ds[2]:-ips[2], ds[1]:-ips[1], ds[0]:-ips[0]]

            outimage = sitk.GetImageFromArray(patchimagearray.astype(np.int8))
            outimage.SetOrigin(image.GetOrigin())
            outimage.SetSpacing(image.GetSpacing())
            outimage.SetDirection(image.GetDirection())
            sitk.WriteImage(outimage, outfile, True)
            arr_path.append(outfile+"\n")

            print("done")
            j = j + 1

fo = open(args.listpath, "w")
fo.writelines(arr_path)
fo.close()

