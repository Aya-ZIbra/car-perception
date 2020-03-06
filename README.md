# Full-stack self-driving car perception
This self driving car uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit and the Intel® Deep Learning Deployment Toolkit. It is designed for a car mounted camera system that detects the driving lane and other vehicle along the road. It is intended to provide real-time car perception, mainly localization, to the self-driving car control system.
## Check the Input and Output video
* Input: Resources/[project\_video.mp4](https://github.com/Aya-ZIbra/car-perception/blob/master/Resources/project_video.mp4)
* Output: results/demo/[road.mp4](https://github.com/Aya-ZIbra/car-perception/blob/master/results/demo/road.mp4)
Check the output vidoe on youtube: https://youtu.be/LpsaYcHkjP4

## Overview of how it works
At start-up the sample application reads the equivalent of command line arguments and loads a network and image from the video input to the Inference Engine (IE) plugin. A job is submitted to an edge compute node with a hardware accelerator such as Intel® HD Graphics GPU, Intel® Movidius™ Neural Compute Stick 2 and Intel® Arria® 10 FPGA. After the inference is completed, the output videos are appropriately stored in the /results/[device] directory, which can then be viewed within the Jupyter Notebook instance.

### How to run the project?
Check the Juptyer Notebook for details. [Self\_Driving\_Car\_Demo.ipynb](https://github.com/Aya-ZIbra/car-perception/blob/master/Self_Driving_Car_Demo.ipynb)
### What are the various command line parameters which could be changed and how to change them. 
The inference code is already implemented in **CarPerception.py**.
The Python code takes in command line arguments for video, model etc.

**Command line argument options and how they are interpreted in the application source code**
```
usage: CarPerception.py [-h] -m MODEL -rm ROADMODEL -i INPUT
                        [-l CPU_EXTENSION] [-d DEVICE] [-o OUTPUT_DIR]
                        [-n NUM_REQUESTS]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to an .xml file with a pre-trained vehicle
                        detection model
  -rm ROADMODEL, --roadmodel ROADMODEL
                        Path to an .xml file with a pre-trained model road
                        segmentation model
  -i INPUT, --input INPUT
                        Path to video file or image.
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers. Absolute path to
                        a shared library with the kernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on; CPU, GPU, FPGA
                        or MYRIAD is acceptable. Looksfor a suitable plugin
                        for device specified(CPU by default)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Name of output directory
  -n NUM_REQUESTS, --num_requests NUM_REQUESTS
                        number of async requests
```
**The description of the arguments used in the argument parser:**
* -m location of the pre-trained IR model which has been pre-processed using the model optimizer. There is automated support built in this argument to support both FP32 and FP16 models targeting different hardware for vehicle detection
* -rm location of the pre-trained IR model which has been pre-processed using the model optimizer. There is automated support built in this argument to support both FP32 and FP16 models targeting different hardware for road segmentation

* -i location of the input video stream

* -o location where the output file with inference needs to be stored (results/[device])
* -d type of Hardware Acceleration (CPU, GPU, MYRIAD, HDDL or HETERO:FPGA,CPU)
* -n Number of inference requests running in parallel
## Demonstration objectives

This is a demo project implementation to find the inference delay using Intel Hardware/Software combination.

This demo showcases:
#### Video input support and pre-processing for different models using OpenCV
#### Multiple model deployment
Vehicle detection and road segmentation
#### Inference performance on edge hardware
Inference performend using OpenVINO on an edge compute node with a hardware accelerator such as Intel® HD Graphics GPU, Intel® Movidius™ Neural Compute Stick 2 and Intel® Arria® 10 FPGA
#### Async API in action
Improving the overall frame-rate of the application by not waiting for the inference to complete but continuing to do things on the host while inference accelerator is busy.
#### Output processing using OpenCV
- **Vehicle detection** Bounding boxes, labels, box colors, confidence threshold, etc.
- **Road Segmentation** Semantic mask for the road (purple)

#### Lane detection
Based on the road marks detected by road segmenation model, an image processing flow for lane detection is implemented using OpenCV.

[Details in demo_util.py]

#### Visualization
Visualizing the road mask, the vehicles' bounding boxes and detected lane lines.
