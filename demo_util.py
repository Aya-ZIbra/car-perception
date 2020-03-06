import numpy as np
import cv2
from argparse import ArgumentParser
def args_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Path to an .xml file with a pre-trained"
                        " vehicle detection model")
    parser.add_argument("-rm", "--roadmodel", required=True,
                        help="Path to an .xml file with a pre-trained model"
                        " road segmentation model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image."
                        )
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute "
                        "path to a shared library with the kernels impl.")
    parser.add_argument("-d", "--device", default="CPU", type=str,
                        help="Specify the target device to infer on; "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                        "for a suitable plugin for device specified"
                        "(CPU by default)")
    #parser.add_argument("-c", "--confidence", default=0.5, type=float,
                        #help="Probability threshold for detections filtering")
    parser.add_argument("-o", "--output_dir", help = "Name of output directory", type = str, default = ".")
    parser.add_argument("-n", "--num_requests", default=2, type=int,
                        help="number of async requests")

    return parser

def createVideo(cap, outfile):
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('A','V','C','1'), fps, (width,height))
        return out    

def draw_boxes(frame, result, thres,  cap, labels_map = None):
    '''
    Draw bounding boxes onto the frame.
    '''
    
    width = int(cap.get(3))
    height = int(cap.get(4))
    thres = float(thres)
    
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        class_id = int(box[1])
        color = (max(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
        if conf >= thres:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            # add labels
            det_label = labels_map[class_id] if labels_map else str(class_id)
            cv2.putText(frame, det_label + ' ' + str(round(box[2] * 100, 1)) + ' %', (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

    return frame

def add_semantic_mask(frame, results, cap):
    width = int(cap.get(3))
    height = int(cap.get(4))
    c, H, W = results.shape[1:] #(1, 4, 512, 896) 
    classmap = np.argmax(results[0], axis = 0)
    assert(classmap.shape == (H,W)) #(512, 896)
    # Values of mask range from 0 -> 3
    # Colors array
    colors= np.vstack([[0, 0, 0],[255, 0, 255],[0, 0, 0], [255, 0, 255] ]).astype("uint8") # (4, 3) bgr
    classmap = cv2.resize(classmap.astype("uint8"),(width,height))
    mask = colors[classmap] # (512, 896, 3) 
    #assert(mask.shape == (H,W,3))
    #mask = cv2.resize(mask, (width,height))
    assert(mask.shape == frame.shape)
    frame =  cv2.addWeighted(frame, 0.8,mask, 0.2, 0)
    
    return frame , classmap


def detect_lanes(classmap, frame):
    mark_mask = np.where(classmap == 3,255,0).astype("uint8") # (720, 1280)
    #road_mask = np.where(classmap == 1,255,0).astype("uint8")
    height, width = mark_mask.shape
    kernel_size = 7 #check effect of kernel_size on number of lines
    mark_mask = cv2.GaussianBlur(mark_mask, (kernel_size, kernel_size), 0)
    
    # 2 methods: use of ROI or cluster lines based on slope (to be tried)
    
    ## ROI method
    vertices = [(0, height),(width / 2, height / 2),(width, height),]#.astype("int32")
    cropped_Image = region_of_interest(mark_mask, vertices)
    cropped_Image = cv2.Canny(cropped_Image, 50,150)
    #cv2.imwrite("outputs/ROI.png", cropped_Image)
    ## HoughLines parameters (I don't fully understand them)
    rho = 2
    theta = np.pi/180
    threshold = 100
    #lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=100, maxLineGap=180) #(19, 1, 4)
    lines = cv2.HoughLinesP(cropped_Image,rho, theta, threshold, np.array ([]), minLineLength=40, maxLineGap=25) # w/o canny (58, 1, 4)
    #print(lines.shape)
    
    ## Group lines
    right_line_pts= []
    left_line_pts = []
    if lines is not None:
        for line in lines:
            line = line.reshape(4)
            slope = get_slope(line)
            #print(line, slope)
            if slope > 0.5:
                right_line_pts.extend((tuple(line[:2]),tuple(line[2:])))
            elif slope< -0.5:
                left_line_pts.extend((tuple(line[:2]),tuple(line[2:])))
            else:
                pass
    #print(right_line_pts)
    pR, y_upperR = poly_fit(right_line_pts, height)
    pL, y_upperL = poly_fit(left_line_pts, height)
    y_lower = height #max(y_pts) 
    #y_upper = min(y_upperL, y_upperR)
    y_upper = 2*height //3
    lines =np.array([get_line(pR,y_lower, min(y_upperR, y_upper)), get_line(pL,y_lower, min(y_upperL, y_upper))])
    #print(lines)
    line_image = display_lines(frame, lines)
    #cv2.imwrite("outputs/lines.png", line_image)
    
    frame =  cv2.addWeighted(frame, 0.6,line_image.astype("uint8"),1,0)
    return frame


def get_slope(line): 
    x1, y1, x2, y2 = line
    if x1 == x2: return 0
    return -(y2-y1)/(x2-x1)

def poly_fit(line_pts,h, deg=1):
    if len(line_pts) ==0:
        return None, h
    x_pts = [pt[0] for pt in line_pts]
    y_pts = [pt[1] for pt in line_pts]
    co_eff = np.polyfit(y_pts, x_pts, deg)
    p = np.poly1d(co_eff)
    y_upper = min(y_pts)
    return p, y_upper
    
def get_line(p, y_upper,y_lower):
    if p == None:
        return None
    x_lower = int(p(y_lower))
    x_upper = int(p(y_upper))
    return [x_lower, y_lower, x_upper, y_upper]
    
def region_of_interest(img,vertices):
    mask = np.zeros_like(img)  
    ignore_mask_color = 255
    vertices = np.array([vertices],).astype("int32")
     #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            x1, y1, x2, y2 = line #.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    return line_image



def draw_heatmap(frame, result, thres, cap, labels_map = None):
    w = int(cap.get(3))
    h = int(cap.get(4))
    feature_maps = result[0][1:]
    i = 1
    for feature_map in feature_maps:
        #cv2.imwrite("outputs/{}-output.png".format(i), feature_map)
        feature_map = cv2.resize(feature_map,(w,h))
        feature_map= np.where(feature_map>thres, 255, 0)
        
        #feature_map = feature_map*255
        feature_mask = get_mask(feature_map,i)
        #cv2.imwrite("outputs/{}-output.png".format(i), frame + feature_mask)
        #print(i)
        i = i+1
        assert (frame.shape == feature_mask.shape)
        #the image you read from disc is CV_8U, and you try to add it to a CV_64F image
        #convert frame: image = np.asarray(image, np.float64)
        # OR add(mask, image, dtype=cv2.CV_64F)
        #frame =  cv2.addWeighted(frame, 0.7, feature_mask.astype("uint8"), 0.3, 0) #, type=cv2.CV_64F)
        frame =  cv2.add(frame, feature_mask.astype("uint8"))
    return frame
def get_mask(processed_output,i):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
    # Create an empty array for other color channels of mask
    #empty = np.zeros(processed_output.shape)
    # Stack to make a Green mask where text detected
    R = processed_output* (1 if i==1 else 0.1)
    G = processed_output* (1 if i==2 else 0.1)
    B = processed_output* (1 if i==3 else 0.1)
    mask = np.dstack((R, G, B))

    return mask

  