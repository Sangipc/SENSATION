from Test_on_videos import segmentation
from depthmap_video import depth_estimation
import sys
def main():
    final_folder = sys.argv[1]  #name a folder as Final_Folder and send the path
    model_path = sys.argv[2]

    status = segmentation(final_folder,model_path)
    depth_status = False
    if status:
        depth_status = depth_estimation(final_folder)

    if depth_status:
        print("Completed Successfully")




if __name__ == "__main__":
    main()
