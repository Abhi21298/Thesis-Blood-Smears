#!/bin/bash -e

cd /home/amr1/Documents/Thesis-Blood-Smears
source activate thesis
rm -rf ../gui_predict
#python -u GUI_args.py --input_image "/home/amr1/Documents/Healthy_BF_Sample 1.tif" > /home/amr1/Documents/Healthy_BF1_cropthresh.log
#mv ../gui_predict ../gui_predict_HBF1_cropthresh
#python -u GUI_args.py --input_image "/home/amr1/Documents/Healthy_BF_Sample 2.tif" > /home/amr1/Documents/Healthy_BF2_cropthresh.log
#mv ../gui_predict ../gui_predict_HBF2_cropthresh
python -u GUI_args.py --input_image "/home/amr1/Documents/Healthy_BF_Sample 3.tif" > /home/amr1/Documents/Healthy_BF3_cropthresh.log
mv ../gui_predict ../gui_predict_HBF3_cropthresh
#python -u GUI_args.py --input_image "/home/amr1/Documents/Healthy_BF_Sample 4.tif" > /home/amr1/Documents/Healthy_BF4_cropthresh.log
#mv ../gui_predict ../gui_predict_HBF4_cropthresh
