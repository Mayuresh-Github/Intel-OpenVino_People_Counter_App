## Project Writeup
This document address several answers to questions for my People Counter App Project. This documents also contains the models I used for this project and how I choosed the best model.

**What are Custom Layers?**
 - Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer      classifies them as custom.
 - One factor behind deep learning’s success is the availability of a wide range of layers that can be composed in creative ways to design architectures suitable for a wide          variety of tasks. For instance, researchers have invented layers specifically for handling images, text, looping over sequential data, performing dynamic programming, etc.        Sooner or later, you will encounter (or invent) a layer that does not exist yet in the framework. In these cases, you must build a custom layer.
 - Check the tutorials for Creating Custom layers by Intel Openvino here: [Link to tutorial](https://github.com/david-drew/OpenVINO-Custom-Layers/tree/master/2019.r2.0)
 - In Tensorflow we develop our own Custom layer by extending the ```tf.keras.Layer class```.
 - Check the tutorials for creating Custom Layers in Tensorflow here: [Link to tutorial](https://www.tensorflow.org/tutorials/customization/custom_layers)
 
---

## How to create Custom Layers using Openvino

- When implementing a custom layer for the model, we need to add extensions to both the Model Optimizer and the Inference Engine.
- Different supported framework has different step for registering custom layers.
- For register custom layers your self follow steps given in this link:- https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md
- When you load IR in Infernece Engine then there may be possibility to found unsuported layers and for that cpu extension can be used. cpu extension in linux:-                     /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_xxxx.so 

---

## Models used for this Project

- Model 1: [Ssd_inception_v2_coco]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - I converted the model to an Intermediate Representation and checked for performance.
  - This model lacked accuracy as it didn't detect people correctly in the video. 
  - The model was insufficient for the app because it failed to detect people in some intervals and this affected accuracy.
  
- Model 2: [Faster_rcnn_inception_v2_coco_2018_01_28]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - I converted the model to an Intermediate Representation and checked for performance.
  - This model performed really well in the output video. The model works better than the previous approaches.
  - After managing the shape attribute it worked quite well.
    
---
  
## Performance Statistics

Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_coco in terms of latency and memory, several insights were drawn. It could be clearly seen that the Latency (microseconds) and Memory (Mb) decreases in case of OpenVINO as compared to plain Tensorflow model which is very useful in case of OpenVINO applications.

| Model/Framework                             | Latency (microseconds)            | Memory (Mb) |
| -----------------------------------         |:---------------------------------:| -------:|
| ssd_inception_v2_coco (plain TF)            | 229                               | 538    |
| ssd_inception_v2_coco (OpenVINO)            | 150                               | 329    |
| faster_rcnn_inception_v2_coco (plain TF)    | 1279                              | 562    |
| faster_rcnn_inception_v2_coco (OpenVINO)    | 891                               | 281    |

So by seeing the above results we can clearly state that faster_rcnn_inception_v2_coco (Openvino) model outperformed from all other models and can detect people with high accuracy. Hence I choosed faster_rcnn_inception_v2_coco_2018_01 model for my Project.

---

## Assess Model Use Cases

Some potential use cases of this Project in Real World
  1. Identify people in Office and Record Attendance and also movement of the Employees
    - By chaining this model with Face Recognition we can use this as a Attendance Marker at Entry door in Companies. Also we can detect movement of employess and this can             provide answers to many questions like: What employees do most of the time, are the employees really working etc.
    
  2. Social Distancing and Alarming 
    - We can use this Project in a open place where we can monitor people and distance between them. We can also put a recorded alarm and when the System detects that people are       violating social distancing we can autoplay the alarm with loud noise and alert people. This can save time of Cops and even we can stop spread of Covid-19. 
    
  3. Intrusion Detection in Companies 
    - As stated in Eg 1 we can chain the model with facial recognition and record all the employees and staff in a Company. We can make a Database of stored faces and put a           Camera at the entry door and when the System will detect known people the door will be automatically opened. This will prevent intruders from entering the Company.

What we discussed as use cases of this Project is just the 'Tip of the Iceberg'. We can use this project in many scenarios.

---

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. 
The potential effects of each of these are as follows : 
 - Poor quality lighting can reduce the accuracy and produce poor quality of result because, for computer vision application the image is essential.
 - The image size/focal length also have effect because the models have inputs requirement(inputs size of images).

---

## Demo run video 
[Link](https://drive.google.com/file/d/1DtQzUcmYtl7xnudc_R_XxppOigrpYtfh/view?usp=sharing)
