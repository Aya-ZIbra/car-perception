import logging as log
from openvino.inference_engine import IECore, IENetwork
import cv2
class DNN:
    '''
    Load and store information for working with the Inference Engine,
    and any loaded models.
    '''
    def __init__(self):
        self.ie = None
        self.net = None
        self.exec_net = None
        self.device = 'CPU'
        self.input_blob = None
        self.out_blob = None
    
    # Create Plugin for specified device and load extensions library if specified
    def createPlugin(self, device, cpu_extension = None):
        log.info("Initializing plugin for {} device...".format(device))
        ie = IECore()
        # Add a CPU extension, if applicable
        if cpu_extension and 'CPU' in device:
            log.info("Loading plugins for {} device...".format(device))
            ie.add_extension(cpu_extension, "CPU")
        return ie
    
    # Create Network and Load the optimized model into it 
    def createNetwork(self, model_xml, model_bin):
        # Importing network weights from IR models.
        log.info("Reading IR...")
        net = IENetwork(model=model_xml, weights=model_bin)
        
        # Getting the input and outputs of the network
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        return net
    
    # Load Network into Plugin
    def loadNetwork(self, device, num_requests = 2):
        ie = self.ie
        net = self.net
        # Check all layers in IR models are supported by device plugins. 
        if 'CPU' in device:
            supported_layers = ie.query_network(net, device)
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                print("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
        
        exec_net = ie.load_network(network=net, num_requests=num_requests, device_name=device)
        return exec_net
    
    # Run inference
    def run_async_inference(self, my_request_id, pframe):
        exec_net = self.exec_net
        exec_net.start_async(request_id=my_request_id, inputs={self.input_blob: pframe})
    
    def run_sync_inference(self, pframe):
        exec_net = self.exec_net
        return exec_net.infer({self.input_blob: pframe})
        
    
    def is_complete(self, my_request_id):
        return self.exec_net.requests[my_request_id].wait(-1)
    def get_output(self, my_request_id):
        exec_net = self.exec_net
        return exec_net.requests[my_request_id].outputs[self.out_blob]
    
    # input preprocessing
    def preprocessInput(self, frame):
        net = self.net
        input_blob = self.input_blob
        n, c, h, w = net.inputs[input_blob].shape
        pframe = cv2.resize(frame, (w, h))   # resize
        pframe = pframe.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        pframe = pframe.reshape((n, c, h, w)) # reshape to add batch size (1, )
        return pframe
    
    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.net
        del self.ie
        del self.exec_net
    #def postprocessOutput(self, result):
        