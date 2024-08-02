import os
import cv2
import re
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import albumentations as albu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dictionaries to store intersection points for left and right points
left_intersection_dict = {}
right_intersection_dict = {}


# def getUserArguments():
#     final_folder = sys.argv[1]
#     model_path = sys.argv[2]
#     filename = sys.argv[3]
#
#     return final_folder, model_path, filename
#
# final_folder,model_path, filename = getUserArguments()

def segmentation(final_folder, model_path):
    class Dataset(BaseDataset):
        """TrayDataset. Read images, apply augmentation and preprocessing transformations.

        Args:
            images_dir (str): path to images folder
            masks_dir (str): path to segmentation masks folder
            class_values (list): values of classes to extract from segmentation mask
            augmentation (albumentations.Compose): data transfromation pipeline
                (e.g. flip, scale, etc.)
            preprocessing (albumentations.Compose): data preprocessing
                (e.g. noralization, shape manipulation, etc.)

        """

        def __init__(
                self,
                images_dir,
                masks_dir,
                classes,
                augmentation=None,
                preprocessing=None,
        ):
            # get images(x) and masks(y) ids
            predout_lst = os.listdir(images_dir)

            def sort_key(filename):
                # Extract the numerical part of the filename using regular expression
                match = re.search(r'\d+', filename)
                if match:
                    return int(match.group())
                return -1  # Return a default value for filenames without numbers

            predout_lst.sort(key=sort_key)

            self.ids_x = predout_lst
            # ['1001a01.jpg', '1005a.jpg', '1006a72.jpg', '2001a72.jpg', '2002a.jpg'] etc.
            predout_mask = os.listdir(masks_dir)

            predout_mask.sort(key=sort_key)

            self.ids_y = predout_mask

            # get images(x) and masks(y) full paths (fps)
            self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_x]
            # '/content/drive/My Drive/Colab Notebooks/TrayDataset/XTest/1001a01.jpg'
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_y]

            # convert str names to class values on masks
            self.class_values = [CLASSES.index(cls.lower()) for cls in classes]
            self.augmentation = augmentation
            self.preprocessing = preprocessing

        def __getitem__(self, i):

            # read data
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)

            # extract certain classes from mask (e.g. cars)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
            # mask = np.stack(masks, axis=-1)

            # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            return image, mask

        def __len__(self):
            return len(self.ids_x)

    CLASSES = ['bird', 'ground animal', 'curb', 'fence', 'guard rail', 'barrier', 'wall', 'bike lane',
               'crosswalk - plain',
               'curb cut', 'parking', 'pedestrian area', 'rail track', 'road', 'service lane', 'sidewalk', 'bridge',
               'building',
               'tunnel', 'person', 'bicyclist', 'motorcyclist', 'other rider', 'lane marking - crosswalk',
               'lane marking - general',
               'mountain', 'sand', 'sky', 'snow', 'terrain', 'vegetation', 'water', 'banner', 'bench', 'bike rack',
               'billboard',
               'catch basin', 'cctv camera', 'fire hydrant', 'junction box', 'mailbox', 'manhole', 'phone booth',
               'pothole',
               'street light', 'pole', 'traffic sign frame', 'utility pole', 'traffic light', 'traffic sign (back)',
               'traffic sign (front)',
               'trash can', 'bicycle', 'boat', 'bus', 'car', 'caravan', 'motorcycle', 'on rails', 'other vehicle',
               'trailer',
               'truck', 'wheeled slow', 'car mount', 'ego vehicle', 'unlabeled']

    def get_training_augmentation():
        train_transform = [

            albu.Resize(256, 320, p=1),
            albu.HorizontalFlip(p=0.5),

            albu.OneOf([
                albu.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=1),
                albu.CLAHE(p=1),
                albu.HueSaturationValue(p=1)
            ],
                p=0.9,
            ),

            albu.IAAAdditiveGaussianNoise(p=0.2),
        ]
        return albu.Compose(train_transform)

    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(256, 320)
        ]
        return albu.Compose(test_transform)

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')

    def get_preprocessing(preprocessing_fn):
        _transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor, mask=to_tensor),
        ]
        return albu.Compose(_transform)

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax'
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION)
    # Normalize your data the same way as during encoder weight pre-training
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Load the model on the CPU
    Trained_model = torch.load(model_path, map_location=DEVICE)

    '''-----------------------------function to find intersection points of sidewalk-------------------------------'''

    def find_intersection_points(image, target_rgb, y_line):
        """
        Find the intersection points of the object with given RGB and the horizontal line at y_line.

        Parameters:
        - image: A numpy array of shape (height, width, 3) representing an RGB image.
        - target_rgb: A tuple (R, G, B) representing the RGB values of the target object (sidewalk).
        - y_line: The y-coordinate of the horizontal line.

        Returns:
        - A tuple of two (x, y) coordinates: the leftmost and rightmost intersections.
        """

        # # Extract the row corresponding to the y_line
        row = image[y_line]

        # Find all x coordinates where the row's color matches target_rgb
        matching_xs = np.where(np.all(row == target_rgb, axis=1))[0]

        # If no matching x coordinates are found, return None for both left and right
        if len(matching_xs) == 0:
            return (0, y_line), (0, y_line)

        # The leftmost intersection is the minimum x coordinate, and the rightmost is the maximum x coordinate
        left_point = (matching_xs[0], y_line)
        right_point = (matching_xs[-1], y_line)

        return left_point, right_point

    def decode_segmentation_map(j, image, classesLength=66):
        Class_label_colors = np.array([
            # Background,     tray        cutlery
            (165, 42, 42),
            (0, 192, 0),
            (196, 196, 196),
            (190, 153, 153),
            (180, 165, 180),
            (90, 120, 150),
            (102, 102, 156),
            (128, 64, 255),
            (140, 140, 200),
            (170, 170, 170),
            (250, 170, 160),
            (96, 96, 96),
            (230, 150, 140),
            (128, 64, 128),
            (110, 110, 110),
            (244, 35, 232),
            (150, 100, 100),
            (70, 70, 70),
            (150, 120, 90),
            (220, 20, 60),
            (255, 0, 0),
            (255, 0, 100),
            (255, 0, 200),
            (200, 128, 128),
            (255, 255, 255),
            (64, 170, 64),
            (230, 160, 50),
            (70, 130, 180),
            (190, 255, 255),
            (152, 251, 152),
            (107, 142, 35),
            (0, 170, 30),
            (255, 255, 128),
            (250, 0, 30),
            (100, 140, 180),
            (220, 220, 220),
            (220, 128, 128),
            (222, 40, 40),
            (100, 170, 30),
            (40, 40, 40),
            (33, 33, 33),
            (100, 128, 160),
            (142, 0, 0),
            (70, 100, 150),
            (210, 170, 100),
            (153, 153, 153),
            (128, 128, 128),
            (0, 0, 80),
            (250, 170, 30),
            (192, 192, 192),
            (220, 220, 0),
            (140, 140, 20),
            (119, 11, 32),
            (150, 0, 255),
            (0, 60, 100),
            (0, 0, 142),
            (0, 0, 90),
            (0, 0, 230),
            (0, 80, 100),
            (128, 64, 64),
            (0, 0, 110),
            (0, 0, 70),
            (0, 0, 192),
            (32, 32, 32),
            (120, 10, 10),
            (0, 0, 0)
        ])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, classesLength):
            idx = image == l
            r[idx] = Class_label_colors[l, 0]
            g[idx] = Class_label_colors[l, 1]
            b[idx] = Class_label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)

        # Define the y-coordinates of the three horizontal lines
        y_coordinates = [100, 200, 300]

        # Iterate through the y-coordinates and calculate intersection points
        for y_coord in y_coordinates:
            # Calculate the intersection points
            left_point, right_point = find_intersection_points(rgb, (244, 35, 232), y_coord)
            # Check if the dictionaries for this y_coord already exist
            if j not in left_intersection_dict:
                left_intersection_dict[j] = []

            if j not in right_intersection_dict:
                right_intersection_dict[j] = []

            # Append the left and right intersection points to the respective lists
            if left_point is not None and right_point is not None:
                left_intersection_dict[j].append(left_point)
                right_intersection_dict[j].append(right_point)

        image_width = r.shape[1]
        # Drawing on the images
        for y_coord in y_coordinates:
            cv2.line(rgb, (0, y_coord), (image_width, y_coord), (0, 0, 255), 2)

            if j in left_intersection_dict:
                matrix = left_intersection_dict[j]
                for left_x in matrix:
                    cv2.circle(rgb, left_x, 5, (0, 255, 0), 2)  # Draw the left intersection points in green

            if j in right_intersection_dict:
                matrix = right_intersection_dict[j]
                for right_x in matrix:
                    cv2.circle(rgb, right_x, 5, (0, 255, 0), 2)  # Draw the left intersection points in green
        return rgb

    with open('input_file_name.txt', 'r') as file:
        for line in file:
            if 'Input_File_Name' in line:
                input_file_Name = line.split("'")[1]
                break
    # %%time
    # import shutil

    try:
        # import time
        # import os
        # import cv2

        start_time1 = time.time()
        '''--------------------------------------------------------------------------------------------
        ---------------------------------------Resizing Video to input clip----------------------------
        ---------------------------------------------------------------------------------------------'''
        ParentFolder = "Final_Folder"
        if not os.path.exists(ParentFolder):
            # print("does not Exist")
            os.makedirs(ParentFolder)
        folder_path_left = final_folder + "/Output_Video/"
        folder_path_right = final_folder + "/Output_Video/"
        # print(folder_path_left)
        ResizedVideoPath = os.path.join(ParentFolder, "Resized_Video")
        if not os.path.exists(ResizedVideoPath):
            os.makedirs(ResizedVideoPath)

        # Open the video file
        video = cv2.VideoCapture(final_folder + '/Input_Video/' + input_file_Name)
        original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # print(original_width)
        original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not video.isOpened():
            print("Error opening video file")

        # Define the output video dimensions
        width = 864
        height = 480

        # Create a VideoWriter object to write the output video
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        Resized_video1 = cv2.VideoWriter('Final_Folder/Resized_Video/output_video2.mp4', fourcc, fps, (width, height))
        Resized_videopath = 'Final_Folder/Resized_Video/output_video2.mp4'

        while True:
            # Read a frame from the input video
            ret, frame = video.read()

            if not ret:
                break

            # Resize the frame to the output video dimensions
            resized_frame = cv2.resize(frame, (width, height))

            # Write the resized frame to the output video
            Resized_video1.write(resized_frame)

        # Release the video objects and close all windows
        video.release()
        Resized_video1.release()
        # cv2.destroyAllWindows()
        print("Resizing done")
        '''-----------------------------------------------------------------------------------------
        ----------------------------Breaking into frames and saving as images-----------------------
        --------------------------------------------------------------------------------------------'''
        # import time

        start_time = time.time()

        # Set the paths to the video file and the output folder
        video_path = Resized_videopath
        os.makedirs("Final_Folder/Original_Frames")
        frames_folder = 'Final_Folder/Original_Frames'

        # Create the output folder if it does not exist
        os.makedirs(frames_folder, exist_ok=True)

        # Open the video file
        video_file = cv2.VideoCapture(video_path)
        # Initialize a frame counter
        frame_count = 0
        update_interval = 1
        # Loop through the video frames
        while True:
            # Read the next frame
            ret, frame = video_file.read()

            # If there are no more frames, exit the loop
            if not ret:
                break

            # Save the frame as an image file in the output folder
            filename = os.path.join(frames_folder, f'frame_{frame_count:04d}.png')
            cv2.imwrite(filename, frame)

            # Increment the frame counter
            frame_count += 1
            print('\rIterator for Frame Breakup:', frame_count, end='')
            if frame_count % update_interval == 0:
                elapsed_time = time.time() - start_time
                iteration_rate = (frame_count + 1) / elapsed_time
                print('  Iteration rate:', round(iteration_rate, 2), 'frames/second', end=' ')

                # Release the video file
        video_file.release()
        print("\nFrames generated")

        '''------------------------------------------------------------------------------------------------
        -------------------------------------Generating Segmented Output-----------------------------------
        ------------------------------------------------------------------------------------------------'''

        # from PIL import Image
        # import time
        os.makedirs("Final_Folder/Predicted_Output")
        output_path = 'Final_Folder/Predicted_Output'
        # To check the iteration
        start_time = time.time()
        update_interval = 1
        x_test_dir = frames_folder
        y_test_dir = frames_folder
        test_dataset = Dataset(
            x_test_dir,
            y_test_dir,
            augmentation=None,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )
        for j in range(len(test_dataset)):
            image, gt_mask = test_dataset[j]
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            predicted_mask = Trained_model.module.predict(x_tensor)
            predicted_output = torch.argmax(predicted_mask.squeeze(), dim=0).detach().cpu().numpy()
            rgb_map = decode_segmentation_map(j, predicted_output, 65)
            fig, ax = plt.subplots()
            ax.imshow(rgb_map, alpha=0.8)
            ax.axis('off')
            fig.savefig(os.path.join(output_path, f'{j}.png'), bbox_inches='tight')
            plt.close(fig)
            print('\rIterator for Segmentation:', j, end='')
            if j % update_interval == 0:
                elapsed_time1 = time.time() - start_time
                iteration_rate = (j + 1) / elapsed_time1
                print('  Iteration rate:', round(iteration_rate, 2), 'frames/second', end=' ')
        print("value of j", j)
        print("\nPredicted Segmentation Generated")

        # Saves the left and right points of the sidewalk into txt files
        file_path_left = os.path.join(folder_path_left, input_file_Name + '_left_points.txt')
        with open(file_path_left, 'w') as file:
            for key, values in left_intersection_dict.items():
                file.write(f"{key}: {','.join([f'({x}, {y})' for x, y in values])}\n")

        file_path_right = os.path.join(folder_path_right, input_file_Name + '_right_points.txt')
        with open(file_path_right, 'w') as file:
            for key, values in right_intersection_dict.items():
                file.write(f"{key}: {','.join([f'({x}, {y})' for x, y in values])}\n")

        '''--------------------------------------------------------------------------------------------------
        ------------------------------------Combine predicted outputs to a video-----------------------------
        --------------------------------------------------------------------------------------------------'''
        # import cv2
        # import os
        # import shutil

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_folder + '/Output_Video/' + input_file_Name + '_segmentation.mp4', fourcc, fps,
                              (original_width, original_height))
        # Iterate through the images in the folder
        predout_lst = os.listdir("Final_Folder/Predicted_Output")

        def sort_key(filename):
            return int(filename.split('.')[0])

        predout_lst.sort(key=sort_key)
        for file in predout_lst:
            if file.endswith(".png"):
                # Read the image file
                img = cv2.imread(os.path.join("Final_Folder/Predicted_Output", file))
                # Check if the image was read successfully
                if img is not None:
                    # Resize the image to match the video dimensions
                    img = cv2.resize(img, (original_width, original_height))

                    # Write the frame to the video file
                    out.write(img)
                else:
                    print(f"Error: Could not read image file {file}")

        # Release the video writer and close the video file
        out.release()
        dir_path = r"Final_Folder/Predicted_Output"
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path = r"Final_Folder/Resized_Video"
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path = r"Final_Folder/Original_Frames"
        shutil.rmtree(dir_path, ignore_errors=True)
        end_time1 = time.time()
        print("Video generated")
        print(f"Time taken to segment this video: {round(end_time1 - start_time1, 2)} seconds")
        return True
    except Exception as e:
        print(f"An exception occurred: {e}")
        dir_path = r"Final_Folder/Predicted_Output"
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path = r"Final_Folder/Resized_Video"
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path = r"Final_Folder/Original_Frames"
        shutil.rmtree(dir_path, ignore_errors=True)
        print("Video not generated")
