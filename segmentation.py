"""This the SENSATION segmentation class to use it in your project"""

import csv

import cv2
import numpy as np
import onnxruntime
import torch


class Segmentator:
    def __init__(self,
                 input_width:int = 544, 
                 input_height:int = 544,
                 model_path:str = None):
        self.input_width = input_width
        self.input_height = input_height
        self.model_path = model_path
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_labels_from_csv()

        # Define rgb for sidewalk in mask
        self.sidewalk_rgb = [244, 35, 232]
        
    def preprocess_image(self, image_array):
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        return image_array

    def inference(self, image):
        size = (self.input_height, self.input_width)
        image = cv2.resize(image,size )
        image = image.astype(np.float32)
        image = self.preprocess_image(image)
        x_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.DEVICE)

        # Run the ONNX model for segmentation
        ort_inputs = {self.onnx_session.get_inputs()[0].name: x_tensor.detach().numpy().astype(np.float32)}
        ort_outputs = self.onnx_session.run(None, ort_inputs)
        predicted_output = np.argmax(ort_outputs[0][0], axis=0)

        return predicted_output

    def mask_to_rgb(self, mask):
        """Map grayscale values in a mask to RGB values using the provided color map.

        :param mask: 2D numpy array representing a grayscale image segmentation mask
    
        :return: 3D numpy array representing an RGB image
        """
        rgb_image = np.zeros((*mask.shape, 3), dtype=np.uint8)  # Initialize an RGB image with zeros

        # Find pixels that match the grayscale value and set their RGB values in the RGB image
        for gray_value, rgb in self.color_map.items():
            indices = (mask == gray_value)
            rgb_image[indices] = rgb

        return rgb_image

    def load_labels_from_csv(self):
        """Loads the class label id and RGB values from CSV file"""
        class_label_colors = {}

        with open('class_colors.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_label = int(row['class_label'])
                red = int(row['red'])
                green = int(row['green'])
                blue = int(row['blue'])
                class_label_colors[class_label] = [red, green, blue]

            self.color_map = class_label_colors

    def get_sidewalk_rgb(self):
        return self.sidewalk_rgb


if __name__ == "__main__":
    image_path = "test/1000032755.jpg"
    save_path = "test/1000032755.png"
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    model_path = "model_weights/DeepLabV3Plus_resnet50.onnx"
    segmentator = Segmentator(model_path=model_path)
    mask = segmentator.inference(image)
    mask_rgb = segmentator.mask_to_rgb(mask)
    mask_rgb = cv2.resize(mask_rgb, (width, height))
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, mask_bgr)
