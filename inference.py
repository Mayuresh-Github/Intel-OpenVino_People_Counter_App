#!/usr/bin/env python3
""" Defining the Environment """

""" Start of inference.py """

""" Intel Copyrights """
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Author  : Mayuresh Mitkari
Contact : mmayuresh12@gmail.com
Purpose : Written for People Counter App Project which is a part of Intel Edge AI for IOT Udacity Nanodegree 
Month & Year : July 2020 
"""

""" Required imports """
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


""" This class id used for loading and configuring inference plugins for the specified target devices and performs synchronous and asynchronous modes for the specified infer requests. """
class Network:

    def __init__(self):
        """ Initializing the variables """
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, CPU_EXTENSION, DEVICE, console_output= False):
        """ Loding the model """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        """ Checking for supported layers """
        if not is_all_layers_supported(self.plugin, self.network, console_output=console_output):
            self.plugin.add_extension(CPU_EXTENSION, DEVICE)
            
        self.exec_network = self.plugin.load_network(self.network, DEVICE)
        
        """ Getting the input layer """
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):       
        """ Returning the shape of input layer """
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
        return input_shapes

    def exec_net(self, net_input, request_id):
        """ Starting an asynchronous request """
        self.infer_request_handle = self.exec_network.start_async(
                request_id, 
                inputs=net_input)
        return 

    def wait(self):
        """ Waiting for request to be complete """
        status = self.infer_request_handle.wait()
        return status

    def get_output(self):
        """ Extracting and returning the output results """
        out = self.infer_request_handle.outputs[self.output_blob]
        return out
    
def is_all_layers_supported(engine, network, console_output=False):
    """ Checking if all layers are supported and returning True if supported else return False """
    layers_supported = engine.query_network(network, device_name='CPU')
    layers = network.layers.keys()

    all_supported = True
    for l in layers:
        if l not in layers_supported:
            all_supported = False
    return all_supported


""" End of inference.py """