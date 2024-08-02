#!/bin/bash
#SBATCH --job-name=MainJOB         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins

#SBATCH --mail-type=end          # send email when job ends

#SBATCH --mail-user=

#SBATCH --output=/home/hpc/iwi5/iwi5134h/Sidewalk/Scripts/vid.%j.out
module purge
module load python/3.8-anaconda
source env/bin/activate 
python main.py /home/hpc/iwi5/iwi5134h/Sidewalk/Scripts/Final_Folder /home/hpc/iwi5/iwi5134h/Sidewalk/Scripts/Testing_on_videos/DeeplabV3Plus_resnet50.pth
