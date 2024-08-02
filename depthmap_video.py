import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LinearRegression
from scipy.interpolate import RectBivariateSpline
import copy
# Imports
import cv2
#from PIL import Image
import time
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Device assignment based on CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Download MiDaS
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.to('cpu')
midas.eval()
#Input transformational pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform
# def getUserArguments():
#     final_folder = sys.argv[1]
#     filename = sys.argv[2]
#
#     return final_folder,filename
#
# final_folder, filename = getUserArguments()
left_list = []
right_list = []
slope = 0.0
direction = 0

def depth_estimation(final_folder):
    with open('input_file_name.txt', 'r') as file:
        for line in file:
            if 'Input_File_Name' in line:
                input_file_Name = line.split("'")[1]
                break
    #Converting Depth to distance
    depth_scale = 1.0

    def depth_to_distance(depth_value, depth_scale):
      return 1.0/(depth_value*depth_scale)

    # %%time
    #import shutil
    try:
        #import time
        #import os
        #import cv2
        start_time1=time.time()
        '''--------------------------------------------------------------------------------------------
        ---------------------------------------Resizing Video to input clip----------------------------
        ---------------------------------------------------------------------------------------------'''
        ParentFolder = "Final_Folder"
        if not os.path.exists(ParentFolder):
            # print("does not Exist")
            os.makedirs(ParentFolder)

        file_path_left = final_folder + '/Output_Video/' + input_file_Name + '_left_points.txt'
        file_path_right = final_folder + '/Output_Video/' + input_file_Name + '_right_points.txt'

        ResizedVideoPath = os.path.join(ParentFolder, "Resized_Video")
        if not os.path.exists(ResizedVideoPath):
            os.makedirs(ResizedVideoPath)

        # Open the video file
        video = cv2.VideoCapture(final_folder+'/Input_Video/' + input_file_Name)
        original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
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
        print("lets go into depth")

        '''--------------------------------------------------------------------------------------------
        ---------------------------------------Drift Detection---- ------------------------------------
        ---------------------------------------------------------------------------------------------'''
        left_depth_in_distance = {}
        global slope
        right_depth_in_distance = {}
        global direction


        drift_dict = {}

        def linear_regression(data_left,data_right):
            global direction
            # Convert the data to numpy arrays and reshape for sklearn
            X_left = np.arange(len(data_left)).reshape(-1, 1)
            y_left = np.array(data_left)

            X_right = np.arange(len(data_right)).reshape(-1, 1)
            y_right = np.array(data_right)

            # Fit a linear regression model
            model_left = LinearRegression()
            model_left.fit(X_left, y_left)

            model_right = LinearRegression()
            model_right.fit(X_right,y_right)


            # Get the slope of the fitted line
            slope_left = model_left.coef_[0]
            slope_right = model_right.coef_[0]

            # If the slope is positive, the line is increasing. If negative, it's decreasing.
            if slope_left > slope_right:
                print("Drifting towards right")
                direction = 1
            else:
                print("Drifting towards left")
                direction = 2
            # else:
            #     print("Walking straight")
            #     direction = 0

            # Output the slope value
            print(f"The slope of the left line: {slope_left}")
            print(f"The slope of the right line: {slope_right}")

            return direction


        def drift_detection(output, left_lines, right_lines, ind, prev, drift):
            global left_list
            global right_list
            # Split lines and extract coordinates
            split_data_left = [item.split('\n')[0] for item in left_lines]
            split_data_right = [item.split('\n')[0] for item in right_lines]
            data_left = extract_coordinates(split_data_left)
            data_right = extract_coordinates(split_data_right)
            # Creating a spline array of non-integer grid
            h, w = output.shape
            x_grid, y_grid = np.arange(w), np.arange(h)
            spline = RectBivariateSpline(y_grid, x_grid, output)

            def process_data(data, depth_dict):
                for frame_number, coordinates in data:
                    if frame_number == ind:
                        points = [(int(coord[0]), int(coord[1])) for coord in coordinates]
                        if ind not in depth_dict:
                            depth_dict[ind] = []
                        for point in points:
                            cv2.circle(img_original, point, 5, (0, 255, 0), 2)
                            depth = spline(point[1], point[0])
                            numerical_value = depth[0, 0]
                            depth_midas = depth_to_distance(numerical_value, depth_scale)
                            depth_dict[ind].append(round(depth_midas, 3))
            process_data(data_left, left_depth_in_distance)
            process_data(data_right, right_depth_in_distance)
            # this code is for each y point on the left side, we are assigning prev as left point when ind = 0
            updated_drift = drift
            updated_prev = prev
            for i in range(6):
                if ind == 0:
                    if i < 3:
                        updated_prev[i] = left_depth_in_distance[ind][i]
                    else:
                        updated_prev[i] = right_depth_in_distance[ind][i-3]
                else:
                    if ind % 50 == 0:
                        if i < 3:
                            updated_drift[i] = round(updated_prev[i] - left_depth_in_distance[ind][i], 2)
                            updated_prev[i] = left_depth_in_distance[ind][i]
                        else:
                            updated_drift[i] = round(updated_prev[i] - right_depth_in_distance[ind][i-3], 2)
                            updated_prev[i] = right_depth_in_distance[ind][i-3]

            # code to save the values of y=100 in a list
            if ind % 50 == 0 and ind != 0:
                start_frame = max(0, ind - 50)  # Calculate the starting frame
                end_frame = ind  # Calculate the ending frame
                for i in range(start_frame, end_frame):
                    left_list.append(left_depth_in_distance[i][1])
                    right_list.append(right_depth_in_distance[i][1])

                direction = linear_regression(left_list, right_list)
                left_list = []
                right_list = []
            if ind % 50 == 0:
                if ind not in drift_dict:
                    drift_dict[ind] = []
                drift_dict[ind].append(copy.copy(updated_drift))

            return updated_drift, updated_prev

        def extract_coordinates(lines):
            data = []
            for item in lines:
                frame_data = item.split(': ')
                frame_number = int(frame_data[0])
                coordinates_str = frame_data[1]
                coordinates = [tuple(map(int, coord.strip('()').split(','))) for coord in coordinates_str.split('),(')]
                data.append((frame_number, coordinates))
            return data

        '''------------------------------------------------------------------------------------------------
        -------------------------------------Generating Depth map------------------------------------------
        ------------------------------------------------------------------------------------------------'''
        # from PIL import Image
        # import time
        print("I am inside Generating depth Output")
        folder_path = frames_folder
        os.makedirs("Final_Folder/Predicted_Output")
        output_path = 'Final_Folder/Predicted_Output'
        # To check the iteration
        img_names = glob.glob(os.path.join(folder_path, "*"))
        img_names.sort()

        updated_drift = []
        updated_prev = []
        with open(file_path_left, 'r') as file:
            left_lines = file.readlines()
        with open(file_path_right, 'r') as file:
            right_lines = file.readlines()

        for ind, img_name in enumerate(img_names):
            if os.path.isdir(img_name):
                continue
            print("  processing {} ({}/{})".format(img_name, ind + 1, len(img_names)))
            img = cv2.imread(img_name)
            img_original = img.copy()
            # Transform input for midas
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgbatch = transform(img).to('cpu')
            # Make a prediction
            with torch.no_grad():
                prediction = midas(imgbatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
                depth_map = prediction.cpu().numpy()
                depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

            initial = [0] * 6
            drift = [0] * 6
            # code to calculate the difference between two frame depths
            if ind == 0:
                updated_drift, updated_prev = drift_detection(depth_map, left_lines, right_lines, ind, initial, drift)
            else:
                updated_drift, updated_prev = drift_detection(depth_map, left_lines, right_lines, ind, updated_prev, updated_drift)

            k_height = [100, 200, 300]
            for i in range(3):
                hei = k_height[i]
                cv2.putText(img_original, str(left_depth_in_distance[ind][i]), (290, hei), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255),
                            3)
                cv2.putText(img_original, str(right_depth_in_distance[ind][i]), (750, hei), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)

            if direction == 0:
                cv2.putText(img_original, "Walking straight", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            elif direction == 1:
                cv2.putText(img_original, "Drifting towards right", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            else:
                cv2.putText(img_original, "Drifting towards left", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            fig, ax = plt.subplots()
            ax.imshow(img_original, alpha=0.8)
            ax.axis('off')
            fig.savefig(os.path.join(output_path, f'{ind}.png'), bbox_inches='tight')
        # Save the depth in txt file
        merged_depth = {}

        # Iterate through frame numbers
        for frame_number in left_depth_in_distance:
            merged_depth[frame_number] = []

            # Iterate through elements in the lists and merge them
            for left_value, right_value in zip(left_depth_in_distance[frame_number], right_depth_in_distance[frame_number]):
                merged_depth[frame_number].append(left_value)
                merged_depth[frame_number].append(right_value)
        folder_path_left = final_folder+'/Output_Video/'
        file_path = os.path.join(folder_path_left, input_file_Name + 'depth_values.txt')
        with open(file_path, 'w') as file:
            for key, values in merged_depth.items():
                file.write(f"{key}: ")
                for i in range(0, len(values), 2):  # Access elements in pairs
                    if i == 4:
                        file.write(f"   300 - ({values[i]:.3f}, {values[i + 1]:.3f})\n")
                    elif i == 2:
                        file.write(f"   200 - ({values[i]:.3f}, {values[i + 1]:.3f})\n")
                    else:
                        file.write(f"100 - ({values[i]:.3f}, {values[i + 1]:.3f})\n")

        file_drift = os.path.join(folder_path_left, input_file_Name + 'drift.txt')
        with open(file_drift, 'w') as file:
            for key, values in drift_dict.items():
                file.write(f"{key}: {values}\n")
        '''--------------------------------------------------------------------------------------------------
        ------------------------------------Combine predicted outputs to a video-----------------------------
        --------------------------------------------------------------------------------------------------'''
        # import cv2
        # import os
        # import shutil

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_folder+'/Output_Video/'+input_file_Name+'depth_masked.mp4', fourcc, fps, (original_width, original_height))
        # print("Outputvideo file generated")
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
        print(f"Time taken to segment this video: {round(end_time1 - start_time1,2)} seconds")
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
