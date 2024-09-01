# Deep Learning pipeline for blood cell segmentation, classification and counting

## Acknowledgment - 

I thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.

## Prerequisites -

1. Please setup the environment for tensorflow and torch for GPU support. The below steps outlines the process using bash command line and assumes the presence of CUDA 12.x and CuDNN 8.9 drivers installed. Kindly alter the version of pytorch, tensorflow according to the versions of the drivers installed in your system.
   
```bash
conda create -n thesis python=3.10.13
conda activate thesis
pip install numpy pandas scikit-learn matplotlib seaborn
pip install onnx onnxruntime opencv-python pycocotools
pip install segment_anything
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs==8.6.1
pip install tensorflow[and-cuda]
```

2. Once the installations are successfully done, pull/fork this github repository onto to your system and download the following models and placed all of them beside the python scripts.

- Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
- Download [CNN_Classifier](https://drive.google.com/file/d/15vL5UkgOWLVVuaVif6HiBUaeEuzXcrvZ/view?usp=drive_link)
- Download [CNN_finetuned_classifier](https://drive.google.com/file/d/1QDYEOpvdG4XISQFKLT7RocRzSU_7yHlo/view?usp=drive_link)
 
## Execution - 

Run GUI.py for launching GUI based application to run the code or execute GUI_args.py to perform the same action in command line argument. Pass the --input_image "<image-to-examine>" as an argument for the code to work on the image.
