import argparse
import json
import os

import cv2
import numpy as np
import torch

from data_handler import InputHandler, InputType
from segmentation import Segmentator


def depth_from_video(ih: InputHandler,
                     segmentator: Segmentator,
                     midas,
                     transform,
                     separate):
    cap = ih.get_cap()

    depth_maps = []
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    original_size = (frame_width, frame_height)
    is_camera = ih.get_input_type() == InputType.CAMERA
    output_path, file_name = os.path.split(ih.get_output_path())
    video_name = "camera_input.mp4" if is_camera else file_name
        
    out = cv2.VideoWriter(os.path.join(
        output_path,
        video_name),
        cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_map = process_image(frame, midas, transform)

        if separate:
            mask = segmentator.inference(frame)
            depth_map = extract_sidewalk_depth(mask, depth_map)

        depth_maps.append(depth_map.tolist())

        # Write frame to output video
        depth_map = cv2.resize(
            depth_map,
            original_size,
            interpolation=cv2.INTER_LINEAR)
        depth_rgb = cv2.applyColorMap(
            cv2.normalize(depth_map, None, 0, 255,
                          cv2.NORM_MINMAX).astype('uint8'), cv2.COLORMAP_JET)
        out.write(depth_rgb)

    cap.release()
    out.release()

    # Save depth maps as JSON
    output_path, filename = os.path.split(ih.get_output_path())
    with open(os.path.join(output_path, filename + '.json'), 'w') as f:
        json.dump(depth_maps, f)


def save_depth_map(depth, base_name, output_folder, original_size):

    # Resize to orginal size
    depth_resize = cv2.resize(depth, original_size, interpolation=cv2.INTER_LINEAR)

    # Normalize and save depth map
    depth_norm = cv2.normalize(depth_resize, None, 0, 255, cv2.NORM_MINMAX)
    depth_rgb = cv2.applyColorMap(depth_norm.astype('uint8'), cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(output_folder, base_name + '.png'), depth_rgb)
    np.savetxt(os.path.join(output_folder, base_name + '_depth.txt'), depth, fmt='%f')



def extract_sidewalk_depth(mask, depth_map):
    """
    Extracts depth values corresponding to the sidewalk in the image.

    :param mask: A segmented mask image where the sidewalk is labeled with ID 15.
    :param depth_map: A depth map of the same scene.
    :return: A numpy array containing depth values for the sidewalk.
    """

    # Resize mask to match depth map size
    resized_mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a mask where sidewalk pixels are marked
    sidewalk_mask = resized_mask == 15

    # Extract depth values where the sidewalk is located
    masked_depth_map = np.full_like(depth_map, 0, dtype=np.float32)
    masked_depth_map[sidewalk_mask] = depth_map[sidewalk_mask]
    
    # return sidewalk_depth_values
    return masked_depth_map


def process_image(img, model, transform):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transforms and predict depth
    input_tensor = transform(img_rgb).cpu()

    with torch.no_grad():
        depth = model(input_tensor)

    depth = depth.squeeze().cpu().numpy()
    return depth


def main():
    parser = argparse.ArgumentParser(description="Depth Map Estimation")
    parser.add_argument("--input", help="Path to the input folder or video file", required=True)
    parser.add_argument("--output", help="Path to the output folder", required=True)
    parser.add_argument("--model_path", help="Path to the segmentation model weights", required=True)
    parser.add_argument('--separate', action='store_true', help='Separate depth for only sidewalk')
    args = parser.parse_args()

    # Load MiDaS model and transforms
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    transform = torch.hub.load('intel-isl/MiDaS', 'transforms').default_transform

    # Load segmentator
    segmentator = Segmentator(model_path=args.model_path)

    ih = InputHandler(
        input_path=args.input,
        output_path=args.output)

    if ih.get_input_type() == InputType.IMAGES:
        for image_path in ih.get_images():
            img = cv2.imread(image_path)
            frame_height, frame_width = img.shape[:2]
            original_size = (frame_width, frame_height)
            depth = process_image(img, midas, transform)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Only sidewalk depth
            if args.separate:
                mask = segmentator.inference(img)
                depth = extract_sidewalk_depth(mask, depth)

            save_depth_map(depth, base_name, args.output, original_size)
    if ih.get_input_type() == InputType.VIDEO:        
        depth_from_video(ih, segmentator, midas, transform, args.separate)


if __name__ == "__main__":
    main()
