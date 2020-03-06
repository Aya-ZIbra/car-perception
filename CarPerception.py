from DNN import DNN
from demo_util import *
import os
import cv2
import numpy as np
import time


#import sys
#import json

#from pathlib import Path
#import logging as log
from qarpo.demoutils import *

args = args_parser().parse_args()
print(args)
device = "CPU" #args.device
ext = None #args.cpu_extension
#/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/

# Vehicle detection instantiation
#model_path = 'models/intel/vehicle-detection-adas-0002/FP32/'
#model_xml = model_path + 'vehicle-detection-adas-0002.xml'
model_xml = args.model
model_bin = os.path.splitext(model_xml)[0] + ".bin"

VD = DNN()
VD.ie = VD.createPlugin(device, ext)
VD.net = VD.createNetwork(model_xml, model_bin)
VD.exec_net = VD.loadNetwork( device, args.num_requests )

#print(VD.input_blob, VD.out_blob)
#print(net.inputs[input_blob].shape)

# rs instantiation
#model_path = 'models/intel/road-segmentation-adas-0001/FP32/'
#model_xml = model_path + 'road-segmentation-adas-0001.xml'
model_xml = args.roadmodel
model_bin = os.path.splitext(model_xml)[0] + ".bin"

RS = DNN()
RS.ie = RS.createPlugin(device, ext)
RS.net = RS.createNetwork(model_xml, model_bin)
RS.exec_net = RS.loadNetwork(device, args.num_requests)

number_infer_requests = args.num_requests
my_request_id = 0
#img_path = 'test1.jpg'
img_path = args.input 
assert os.path.isfile(args.input), "Specified input file doesn't exist"

cap = cv2.VideoCapture(img_path)
cap.open(img_path)

video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(video_len)  #1260

if video_len < number_infer_requests:
        number_infer_requests = video_len 
out = createVideo(cap, os.path.join(args.output_dir, "road.mp4"))#'output_test.mp4')


# for the progress bar
job_id = os.environ['PBS_JOBID']
progress_file_path = os.path.join(args.output_dir,'i_progress_'+str(job_id)+'.txt')

infer_time_start = time.time()
#delay
DELAY = 1000 / cap.get(cv2.CAP_PROP_FPS)

#average delay per frame
det_time_VD = []
Seg_time_RS = []


## flags used in Asnyc masking
current_inference = 0
previous_inference = 1 - number_infer_requests
frame_count = 0
required_inference_requests_were_executed = False
Flag_stream_end = 0
process_residual = 0
frames =[0 for i in range(number_infer_requests)]
while cap.isOpened():
    ret, frame = cap.read()
    #print(frame_count, cap.get(cv2.CAP_PROP_POS_FRAMES ), current_inference, previous_inference,Flag_stream_end , process_residual )
    
    if ret == 0:
        Flag_stream_end = 1
    if ret == 1 and frame is None:
        print("checkpoint ERROR! blank FRAME grabbed")
        break
    key_pressed = cv2.waitKey(60)
    if key_pressed == 27:
        break
    if Flag_stream_end ==1:
        process_residual += 1
        if process_residual > number_infer_requests-1:
            assert previous_inference == current_inference, "not all frames are post processed"
            break
    if  Flag_stream_end == 0:
        # Read next frame from input stream if available and submit it for inference 
        my_request_id=current_inference
        # Do preprocessing
        pframe = VD.preprocessInput(frame)
        # run inference
        inf_start_VD = time.time()
        VD.run_async_inference(my_request_id, pframe) 
        # Do preprocessing
        pframe = RS.preprocessInput(frame)
        # run inference
        inf_start_RS = time.time()
        RS.run_async_inference(my_request_id, pframe)
        #print(current_inference)
        frames[current_inference] = frame
        current_inference += 1
        if current_inference >= number_infer_requests:
            current_inference = 0

    
    # Retrieve the output of an earlier inference request
    if previous_inference >= 0:
        my_request_id = previous_inference
        frame = frames[previous_inference]
        if (RS.is_complete(my_request_id) != 0) or (VD.is_complete(my_request_id) != 0) : 
            print (my_request_id, frame_count)
            raise Exception("Infer request not completed successfully")
        if VD.is_complete(my_request_id)== 0: det_time_VD.append(time.time() - inf_start_VD)
        if RS.is_complete(my_request_id)== 0: Seg_time_RS.append(time.time() - inf_start_RS)

    
        #Get inference results
        result = RS.get_output(my_request_id)
        #Post-process: 
        frame, classmap = add_semantic_mask(frame, result, cap)
        frame = detect_lanes(classmap, frame)
        
        #Get inference results
        result = VD.get_output(my_request_id)
        #Post-process: Update the frame to include detected bounding boxes
        frame = draw_boxes(frame, result, 0.5, cap)
        
        #Progress information
        frame_count += 1
        if frame_count%2 == 0: 
            progressUpdate(progress_file_path, int(time.time()-infer_time_start), frame_count, video_len)
        
        # write frame
        if frame_count <10: 
            det_time_VD_avg = det_time_VD[-1]
            Seg_time_RS_avg = Seg_time_RS[-1]
        if frame_count%10 == 0:
            det_time_VD_avg = sum(det_time_VD)/len(det_time_VD)
            Seg_time_RS_avg = sum(Seg_time_RS)/len(Seg_time_RS)
        inf_time_message1 = "Vehicle detection Inference time: {:.3f} ms.".format(det_time_VD_avg * 1000)
        inf_time_message2 = "Road segmentation Inference time: {:.3f} ms.".format(Seg_time_RS_avg * 1000)
        cv2.putText(frame, inf_time_message1, (0, 15), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(frame, inf_time_message2, (0, 35), cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (255, 255, 255), 1)
        out.write(frame)
        
    
    previous_inference += 1
    if previous_inference >= number_infer_requests:
        previous_inference = 0


total_time = time.time() - infer_time_start
print('total_time',total_time, 'frame_count =' , frame_count) 

if args.output_dir:
    with open(os.path.join(args.output_dir, 'stats.txt'), 'w+') as f:
        f.write(str(round(total_time, 1))+'\n')
        f.write(str(frame_count)+'\n')
progressUpdate(progress_file_path, int(time.time()-infer_time_start), frame_count, frame_count)


# 1 : total_time 199.1707580089569 frame_count = 1257, 
# 2 : total_time 176.958074092865 frame_count = 1256
# 3 : total_time 177.98705768585205 frame_count = 1255
out.release()
cap.release()
RS.clean()
VD.clean()