"""
Example Test:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

Example Evaluation:
python test_image.py \
    --images-dir "PATH_TO_IMAGES_DIR" \
    --gt-dir "PATH_TO_GT_ALPHA_DIR" \
    --result-dir "PATH_TO_RESULT_DIR" \
    --pretrained-weight ./pretrained/SGHM-ResNet50.pth

"""

import argparse
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from model.model import HumanSegment, HumanMatting
import utils
import inference
import imutils

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, default="results")
parser.add_argument('--pretrained-weight', type=str, default="./SGHM-ResNet50.pth")

args = parser.parse_args()

if not os.path.exists(args.pretrained_weight):
    print('Cannot find the pretrained model: {0}'.format(args.pretrained_weight))
    exit()

# --------------- Main ---------------
# Load Model
model = HumanMatting(backbone='resnet50')
model = nn.DataParallel(model).cuda().eval()
model.load_state_dict(torch.load(args.pretrained_weight))
print("Load checkpoint successfully ...")


# Load Images
image_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.jpg'), recursive=True),
                    *glob.glob(os.path.join(args.images_dir, '**', '*.jpeg'), recursive=True),
                    *glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)
                    ])

num_image = len(image_list)
print("Find ", num_image, " images")


def get_background_solid(shape):
    # Background Color 
    back_np = np.full(shape, 0)
    color = 255
    back_np[:, :, 0] = color
    back_np[:, :, 1] = color
    back_np[:, :, 2] = color

    return back_np
    

# Process 
for i in range(num_image):
    image_path = image_list[i]
    image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]
    print(i, '/', num_image, image_name)

    with Image.open(image_path) as img:
        img = img.convert("RGB")

    # inference
    pred_alpha, pred_mask = inference.single_inference(model, img)
    print(pred_alpha.shape, pred_mask.shape, np.unique(pred_mask))    
    print(pred_alpha)
    mask = (pred_alpha * 255).astype(np.uint8)     
    mask[mask < 40] = 0
    
    cv2.namedWindow("mask")
    def print_img(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(mask[y,x])
    cv2.setMouseCallback('mask', print_img)

    print(np.unique(mask))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
    print(image.shape)

    print(pred_alpha.shape, np.unique(pred_alpha), pred_alpha.dtype)
    masked = pred_alpha * image + (1 - pred_alpha) * get_background_solid(image.shape)
    masked = masked.astype(np.uint8)        
    # masked = cv2.bitwise_and(image, image, mask=mask)
    masked = imutils.resize(masked, width=800)
    mask = imutils.resize(mask, width=800)
    
    cv2.imshow("image", image)
    cv2.imshow("mask", mask)    
    cv2.imshow("masked", masked)
    if cv2.waitKey(0) == 27:
        exit()

    # save results
    output_dir = args.result_dir + image_path[len(args.images_dir):image_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = output_dir + '/' + image_name + '.png'
    # Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)
    cv2.imwrite(save_path, masked)    
