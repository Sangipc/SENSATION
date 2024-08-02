import argparse
import os
import shutil
import cv2
import numpy as np
from segmentation import Segmentator

# Parameters for Lucas-Kanade optical flow method
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Parameters for feature detection in frames
feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=10,
                      blockSize=7)

trajectory_len = 40  # Length of trajectory to track movement
detect_interval = 10  # Interval to detect new features
trajectories = []  # List to store trajectories
frame_idx = 0  # Initialize frame index counter
movements_track = {}


# Function to perform optical flow on video frames and save processed frames
def optical_flow_video(frames, output_path, segmentator):
    global frame_idx
    global trajectories  # Use the global 'trajectories' variable
    global movements_track

    for frame in frames:  # Loop through each frame in the video
        print("Frame Number: ",frame_idx)
        if frame_idx == 0:  # For the first frame
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            prev_gray = frame_gray  # Store current frame as previous frame
            frame_idx +=1
            continue
        else:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            img = frame.copy()  # Create a copy of the frame

            # Optical flow calculations to track trajectories
            if len(trajectories) > 0:  # Check if trajectories exist
                img0, img1 = prev_gray, frame_gray  # Previous and current frame for optical flow
                p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_trajectories = []
                movements = []

                # Update trajectories based on optical flow and calculate movements
                for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    trajectory.append((x, y))
                    if len(trajectory) > trajectory_len:
                        del trajectory[0]
                    new_trajectories.append(trajectory)
                    cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
                    start_point = np.array(trajectory[0])
                    end_point = np.array(trajectory[-1])
                    movement = end_point - start_point
                    movements.append(movement)
                    movements_track[frame_idx] = movement
                    #Calculating angle
                    if len(trajectory) >= 2:
                        point1 = np.array(trajectory[-2])
                        point2 = np.array(trajectory[-1])
                        displacement_vector = point2 - point1
                        angle = np.degrees(np.arctan2(displacement_vector[1], displacement_vector[0]))
                    # print(movements)
                trajectories = new_trajectories  # Update trajectories with new positions

                # Inside the loop where the text is displayed based on average movement
                if len(movements) > 0:
                    avg_direction = np.mean(movements, axis=0)
                    direction_text = ""
                    if avg_direction[0] > 0:
                        direction_text = "Left"
                    elif avg_direction[0] < 0:
                        direction_text = "Right"

                # Display the text continuously throughout all frames
                cv2.putText(img, f"Angle: {angle:.2f} degrees", (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.putText(img, direction_text, (20, 80), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 6)
                cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))

            # Detect new features at regular intervals
            if frame_idx % detect_interval == 0:
                mask = segmentator.inference(img)

                # Resize mask to match depth map size
                resized_mask = cv2.resize(mask, (frame_gray.shape[1], frame_gray.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Create a mask where sidewalk pixels are marked
                sidewalk_mask = resized_mask == 15
                # mask = np.full_like(frame_gray, 0, dtype=np.float32)
                # mask[sidewalk_mask] = frame_gray[sidewalk_mask]
                mask = np.zeros_like(frame_gray)
                mask[sidewalk_mask] = 255

                for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
                    cv2.circle(mask, (x, y), 5, 0, -1)

                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        trajectories.append([(x, y)])


            prev_gray = frame_gray
            cv2.imwrite(f'{output_path}/{frame_idx}.png', img)  # Save processed frame
            frame_idx += 1

    with open('movements.txt', 'w') as file:
        for frame_idx, movement in movements_track.items():
            file.write(f"movements_track[{frame_idx}] = {movement}\n")

# Main function for processing video and creating output
def main():
    parser = argparse.ArgumentParser(description="Drift detection with Optical Flow")
    parser.add_argument("--input", help="Path of the video file", required=True)
    parser.add_argument("--output", help="Path to the output folder", required=True)
    parser.add_argument("--model_path", help="Path to the segmentation model weights", required=True)
    parser.add_argument('--separate', action='store_true', help='Separate depth for only sidewalk')
    args = parser.parse_args()  # Parse command-line arguments

    cap = cv2.VideoCapture(args.input)  # Read video file
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get original video width
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get original video height
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second (FPS)
    frame_images = []  # Initialize list to store frames
    current_frame = 0  # Initialize frame counter

    # Read each frame of the video and store in a list
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:  # If frame not read successfully, break the loop
            break
        frame_images.append(frame)  # Append frame to the list
        current_frame += 1  # Increment frame counter

    os.makedirs(args.output + "/Output_frames/", exist_ok=True)  # Create output directory for processed frames
    output_frames_path = args.output + "/Output_frames"  # Path to store processed frames
    segmentator = Segmentator(model_path=args.model_path)
    optical_flow_video(frame_images, output_frames_path, segmentator)  # Apply optical flow to video frames

    # Create a video from processed frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output + '/optical_flow.mp4', fourcc, fps, (original_width, original_height))

    # Sort frames before creating the video
    frames_list = os.listdir(output_frames_path)
    frames_list.sort(key=lambda filename: int(filename.split('.')[0]))

    # Write processed frames to the output video
    for file in frames_list:
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(output_frames_path, file))
            if img is not None:
                img = cv2.resize(img, (original_width, original_height))
                out.write(img)
            else:
                print(f"Error: Could not read image file {file}")

    # Remove temporary directory and its contents
    try:
        shutil.rmtree(output_frames_path, ignore_errors=True)
        print(f"Directory {output_frames_path} successfully deleted.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()  # Execute the main function if the script is run directly
