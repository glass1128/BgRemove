from argparse import ArgumentParser
from rembg.bg import remove as U2NET_Remove_Background
from rembg.bg import new_session as onnx_inference_session
from passport import get_passport_image, detector, get_background_solid, resize

import os
import cv2
import numpy as np
import imutils
import torch
import torch.nn as nn
from model.model import HumanMatting
import utils
import inference
from PIL import Image

# ================== Load Models ===========================
pretrained_weight = "./SGHM-ResNet50.pth"
if not os.path.exists(pretrained_weight):
    print('Cannot find the pretrained model: {0}'.format(pretrained_weight))
    exit()
# Load Model Human Matting
model_matting = HumanMatting(backbone='resnet50')
model_matting = nn.DataParallel(model_matting).eval()
model_matting.load_state_dict(torch.load(pretrained_weight, map_location=torch.device("cpu")))
print("Load checkpoint successfully ...")

# Load Model background removal
onnx_session = onnx_inference_session(model_name="u2net")

# =============================================================

def human_matting(img):
    global model_matting
    # inference
    pred_alpha, pred_mask = inference.single_inference(model_matting, Image.fromarray(img).convert("RGB"))
    # print(pred_alpha.shape, pred_mask.shape, np.unique(pred_mask))    
    mask = (pred_alpha * 255).astype(np.uint8)     
    # mask[mask < 15] = 0
    
    cv2.namedWindow("mask")
    def print_img(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(mask[y,x], " .. at (", x,y, ")" )
    cv2.setMouseCallback('mask', print_img)

    # print(pred_alpha.shape)
    # masked = pred_alpha * img + (1 - pred_alpha) * get_background_solid(img.shape)
    # return masked
    
    mask = mask[:,:,0]
    transparent = np.dstack((img, mask))
    return transparent

def remove_background(path, args, visualize = False):
    # access the same onnx u2net inference session, for better performance180
    global onnx_session

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    transparent = None
    if args.always_matting:    
        transparent = human_matting(image)
    else:        
        face_box = detector.detect_main_face(imutils.resize(image, width=640))        
        # print(face_box)
        if face_box is None or not face_box or len(face_box) == 0:     
            print("no faces run u2net")
            transparent = U2NET_Remove_Background(resize(image), session=onnx_session)
        else:
            print("face detected run human matting")
            transparent = human_matting(image)
        
    
    if visualize:
        transparent_original = transparent.copy()
        transparent = imutils.resize(transparent, width=640)
        image = imutils.resize(image, width=640)
        mask = transparent[:,:,3]
    
        # visualize white
        white = get_white_background(transparent[:,:,0:3], mask)
        cv2.namedWindow("white_background")
        cv2.namedWindow("mask")
        def print_img(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(mask[y,x], " .. at (", x,y, ")" )
        cv2.setMouseCallback('white_background', print_img)
        cv2.setMouseCallback('mask', print_img)

        cv2.imshow("transparent", transparent)
        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("white_background", white)
        
        passport = get_passport_image(transparent_original, args.width_passport, args.aspect_ratio)
        cv2.imshow("Passport", passport)
        if cv2.waitKey(0) == 27:
            exit(0)
        return transparent_original
    else:
        return transparent

def get_white_background(image, mask, color = 255):
    mask_repeated = np.stack([mask, mask, mask], axis=2)
    pred_alpha = mask_repeated.astype(np.float32) / 255.0

    masked = pred_alpha * image + (1 - pred_alpha) * get_background_solid(image.shape, color)

    masked = masked.astype(np.uint8)

    return masked


def save_image(image, path):
    cv2.imwrite(path, image)

def save_images(transparent_image, image_name, folders_paths, args):
    filename, file_extension = os.path.splitext(image_name)
    image_name = filename + ".png"
    for key, path in folders_paths.items():
        save_path = os.path.join(path, image_name)

        if key == "transparent":
            save_image(transparent_image, save_path)

        elif key == "white_background":
            white_background = get_white_background(transparent_image[:,:,0:3], transparent_image[:,:,3])
            save_image(white_background, save_path)
        
        elif key == "passport":
            passport_image = get_passport_image(transparent_image, args.width_passport, args.aspect_ratio)
            save_image(passport_image, save_path)            
            

def create_folders(args):
    if not args.save_path:
        return None
    
    save_path = args.save_path        

    paths = {}
    if args.transparent:
        print(save_path, paths)
        paths["transparent"] = os.path.join(save_path, "transparent")
    if args.white_background:
        paths["white_background"] = os.path.join(save_path, "white_background")
    if args.passport:
        paths["passport"] = os.path.join(save_path, "passport")

    for key, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
    return paths
    

def main(args):        
    if args.image:
        transparent = remove_background(args.image, args, args.visualize)
        if args.save_path:
            folders_paths = create_folders(args)
            image_name = os.path.split(args.image)[1]
            save_images(transparent, image_name, folders_paths, args)

    elif args.folder:
        names = os.listdir(args.folder)
        paths = [os.path.join(args.folder, name) for name in names]
        
        folders_paths = create_folders(args)
        for path, image_name in zip(paths, names):
            # remove the background
            transparent_image = remove_background(path, args, args.visualize)
            
            if folders_paths is not None:            
                # save the transparent, white_background, passport if exists
                save_images(transparent_image, image_name, folders_paths, args)
            print("####################")            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, help="path to image")    
    # parser.add_argument("--folder", type=str, help="path to folder of images")    
    parser.add_argument("--folder", type=str, help="path to folder of images", default="../data_matt")    
    parser.add_argument("--save_path", type=str, help="path to save no-background images")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--passport", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--width_passport", type=int, default=600)
    parser.add_argument("--always_matting", action="store_true")
    parser.add_argument("--aspect_ratio", type=float, default=1.28571429)
    
    args = parser.parse_args()
    main(args)
    
    