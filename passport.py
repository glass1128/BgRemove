from facenet_pytorch import MTCNN
import cv2
import os
import numpy as np
import imutils

WIDTH_FACTOR = 0.6
HEIGHT_FACTOR_TOP = 0.7
HEIGHT_FACTOR_BOTTOM = 0.6

class FaceDetector():
    def __init__(self):
        self.mtcnn = MTCNN()
        # self.mtcnn.to(device)
        # print(self.mtcnn)
        self.probability_threshold = 0.5

    def draw(self, frame, boxes, probs):
        for box, prob in zip(boxes, probs):
            cv2.rectangle(frame,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255),
                            thickness=2)

            cv2.putText(frame, str(
                prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame        
    def inference(self, image, return_probs = False):
            boxes, probs = self.mtcnn.detect(image)
            # print("probs ============", probs)
            if boxes is None or probs is None:            
                return [], []
            filtered_boxes = []
            filtered_probs = []
            for box, prob in zip(boxes, probs):
                if prob < self.probability_threshold:
                    continue
                box = [int(x) for x in box]
                filtered_boxes.append(box)
                filtered_probs.append(prob)
            if return_probs:
                return filtered_boxes, filtered_probs
            else:
                return filtered_boxes    

    def detect_main_face(self, image):
        boxes, probs = self.inference(image, return_probs=True)
        # print("boxes, probs", boxes, probs)
        if len(boxes) == 0:
            return None
        
        max_idx = 0
        max_prob = probs[0]
        for i, prob in enumerate(probs):
            if prob > max_prob:
                max_idx = i
        return boxes[max_idx]
                
    def evaluate(self, images_paths):
        for image_path in os.listdir(images_paths):
            image = cv2.imread(os.path.join(images_paths, image_path), cv2.IMREAD_COLOR)
            # boxes, probs = self.inference(image, return_probs=True)            
            # image = self.draw(image, boxes, probs)
            box = self.detect_main_face(image)
            if box is not None:
                cv2.rectangle(image,(box[0], box[1]),(box[2], box[3]),(0, 0, 255), thickness=2)            
            # cv2.imshow("image", image)
            # if cv2.waitKey(0) == 27:
            #     exit()        

def get_background_solid(shape, color = 255):
    # Background Color     
    back_np = np.full(shape, 0)
    back_np[:, :, 0] = color
    back_np[:, :, 1] = color
    back_np[:, :, 2] = color
    return back_np


def resize(image, shape = (640, 640)):
    width, height = shape        
    resized = imutils.resize(image, width=width)
    h, w = resized.shape[0:2]
    # print(w, h)
    # if width < height (portrait mode)
    if w < h:
        # add border
        border_width = abs(h - w)
        # print(border_width)
        color = (114,114,114)
        # color = (0,0,0)
        # color = (255,255,255)
        resized = cv2.copyMakeBorder(resized, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=color)  
        resized = cv2.resize(resized, shape)

    return resized

detector = FaceDetector()

def get_passport_image(transparent_image, width_resize = 320, aspect_ratio = 1.285):
    global detector
    
    # h, w = transparent_image.shape[0:2]
    # if w < h:    
    #     transparent = resize(transparent_image, (h ,h))
    # else:
    # transparent = transparent_image
    
    # transparent = resize(transparent_image)
    transparent = imutils.resize(transparent_image, width=640)
    
    image = transparent[:,:,0:3]
    mask = transparent[:,:,3]

    passport_image = None
    box = detector.detect_main_face(image)
    if box is None or not box:
        image = transparent_image[:,:,0:3]
        mask = transparent_image[:,:,3]
        passport_image = image
    
    else:    
        x1, y1, x2, y2 = box
                
        width, height = x2 - x1, y2 - y1
        x1_new = int(x1 - width * WIDTH_FACTOR)
        x2_new = int(x2 + width * WIDTH_FACTOR)
        y1_new = int(y1 - height * HEIGHT_FACTOR_TOP)
        y2_new = int(y2 + height * HEIGHT_FACTOR_BOTTOM)

        # print(x1_new, y1_new, x2_new, y2_new)
        
        x1_new = max(0, x1_new)
        y1_new = max(0, y1_new)
        x2_new = min(image.shape[1], x2_new)
        y2_new = min(image.shape[0], y2_new)

        # print(x1_new, y1_new, x2_new, y2_new)
        # print("image.shape", image.shape)            
        
        passport_image = image[y1_new: y2_new, x1_new: x2_new, :]
        mask = mask[y1_new: y2_new, x1_new: x2_new]

    # get light gray background
    light_gray_color = 200
    mask_repeated = np.stack([mask, mask, mask], axis=2)
    pred_alpha = mask_repeated.astype(np.float32) / 255.0
    masked = pred_alpha * passport_image + (1 - pred_alpha) * get_background_solid(passport_image.shape, light_gray_color)
    masked = masked.astype(np.uint8)

    ratio_height = int(masked.shape[1] * (aspect_ratio))
    border_height = int(abs(ratio_height-masked.shape[0]) // 2)
    border_width = border_height // 2
    ratio_image = cv2.copyMakeBorder(masked, border_height, 0, border_width//2, border_width//2, cv2.BORDER_CONSTANT, value=(light_gray_color, light_gray_color, light_gray_color))  
    ratio_image = imutils.resize(ratio_image, width=width_resize)
    
    gray_top = 0
    gray_search_area = int(ratio_image.shape[0] * 0.25)
    for i in range(gray_search_area):
        horizontal_line = ratio_image[i,:, 0]
        gray_line = all(pixel == light_gray_color for pixel in horizontal_line)
        if gray_line:
            gray_top += 1
    
    required_gray = 10
    if gray_top > required_gray:
        print(gray_top)
        ratio_image = ratio_image[gray_top - required_gray:, :,:]    

    return ratio_image
    
if __name__ == "__main__":
    # face_detector = FaceDetector()
    # face_detector.evaluate("data")
    
    root = "resultsss/transparent"
    save_path = "resultsss/xx"
    for image_path in os.listdir(root):
        try:
            image = cv2.imread(os.path.join(root, image_path), cv2.IMREAD_UNCHANGED)
            image = imutils.resize(image, width=640)
            # print(image.shape)
            passport = get_passport_image(image)
            
            cv2.imwrite(os.path.join(save_path, image_path), passport)            
            cv2.imshow("passport", passport)
            # cv2.imshow("image", image)
            if cv2.waitKey(0) == 27:
                exit()        
        except():
            pass