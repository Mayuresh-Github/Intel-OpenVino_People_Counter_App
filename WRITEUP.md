# Project Write-Up

The people counter application is a smart video IoT solution that can detect people in a designated area of observation, providing the number of people in the frame, average duration of people in frame, total count of people since the start of the observation session and an alarm that sends an alert to the UI telling the user when a person enters the video frame. It alerts the user when the total count of people that have entered the video since the start of the observation session is greater than five. It was developed as a project required to graduate the Udacity and Intel AI at the Edge Nanodegree program.

The app makes use of Intel® hardware and software tools for deployment. The people counter app makes use of the Inference Engine included in the Intel® Distribution of openVINO™ Toolkit to run it's Edge computations, inferencing and classification processes. The model results are then filtered by a python script (the main.py file) to only identify people in a video recording, camera feed or image.

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)
* OR Udacity classroom workspace for the related course

### Software

*   Intel® Distribution of OpenVINO™ toolkit 2019 R3 release
*   Node v6.17.1
*   Npm v3.10.10
*   CMake
*   MQTT Mosca server

## Summary of Results

Upon testing the people counter app with videos and images of objects besides people, for example cars, dogs e.t.c., it was discovered that the app did not draw any bounding boxes for images other than those classified as people. All other object classes were ignored except the 'person' class.

In conclusion, the app works as intended and identifies only people present in the video or image. It also performs all calculations and inferences as intended and returns the desired results in the desired formats through the UI accessed via the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) using a web browser or in the local storage at the `outputs\videos\` for video outputs and `outputs\images\` directory for images. A few videos and images have been provided in the `resources` folder located in the home directory to use to the app.

## Model Research

In investigating potential people counter models, I tried each of the following two models:

- Model 1: [Ssd_inception_v2_coco]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - Converted the model to intermediate representation using the following command. Further, this model lacked accuracy as it didn't detect people correctly in the video. 
  - The model was insufficient for the app because when i tested it failed on intervals and it didn't found the bounding boxes around the person and for next person.
  - I converted the model to an Intermediate Representation with the following arguments :
    - ```tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz```
    - ```cd ssd_inception_v2_coco_2018_01_28```
    - ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json```
  
- Model 2: [Faster_rcnn_inception_v2_coco_2018_01_28]
  - [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md]
  - Converted the model to intermediate representation using the following command. it performed really well in the output video. The model works better than the previous approaches.
  - After managing the shape attribute it worked quite well.
  - I converted the model to an Intermediate Representation with the following arguments :
    - ```tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz```
    - ```cd faster_rcnn_inception_v2_coco_2018_01_28.tar.gz```
    - ```python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json```
  
## Comparing Model Performance

Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_coco in terms of latency and memory, several insights were drawn. It could be clearly seen that the Latency (microseconds) and Memory (Mb) decreases in case of OpenVINO as compared to plain Tensorflow model which is very useful in case of OpenVINO applications.

| Model/Framework                             | Latency (microseconds)            | Memory (Mb) |
| -----------------------------------         |:---------------------------------:| -------:|
| ssd_inception_v2_coco (plain TF)            | 229                               | 538    |
| ssd_inception_v2_coco (OpenVINO)            | 150                               | 329    |
| faster_rcnn_inception_v2_coco (plain TF)    | 1279                              | 562    |
| faster_rcnn_inception_v2_coco (OpenVINO)    | 891                               | 281    |


## Model Info

From the above conclusion I found a suitable model that works well for the purpose of our app. Below is info about the model:

Model: `faster_rcnn_inception_v2_coco_2018_01_28`

Framework: `Tensorflow`

Total size: `216MB`

Device type used for conversion: `CPU`

Contents of folder: `model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`, `pipeline.config`, `frozen_inference_graph.pb`, `frozen_inference_graph.mapping`,`checkpoint` and a folder named `saved_model`(which contains the `saved_model.pb` file and an empty folder named `variables`)

Brief Description:
Faster RCNN is an object detection model presented by Ross Girshick, Shaoqing Ren, Kaiming He and Jian Sun in 2015, and is one of the famous object detection architectures that uses convolution neural networks. Fast R-CNN passes the entire image to a ConvNet which generates regions of interest instead of passing the extracted regions from the image like a regular R-CNN. Instead of using three different models LIKE the R-CNN, it uses a single model which extracts features from the regions, classifies them into different classes, and returns the bounding boxes. All these steps are done simultaneously, thus making it execute faster than the R-CNN. Fast R-CNN is, however, not fast enough when applied on a large dataset as it also uses selective search for extracting the regions.

Faster R-CNN fixes the problem by replacing it with the Region Proposal Network (RPN). It would first extract the feature maps from the input image using a ConvNet and then pass them to a RPN which returns object proposals then these maps are classified and the bounding boxes are predicted.

## Explaining Custom Layers

### What are custom layers

Rarely, there are some layers in a neural network model that are not in the openVINO Model Optimizer supported layers list. They are layers that are not natively supported by the openvino Inference engine. These layers can be added to the Inference Engine as custom layers. The custom layer can therefore be defined as any model layer that is not natively supported by the model framework.

Though the custom layer is a useful feature and there can be some unsupported layers in our model. The custom layer feature of the model optimizer is rarely needed because these unsupported layers are usually supported by the built-in device extensions in the openvino tool kit. For example, the unsupported layers in the faster rcnn model are the 'proposals', 'Squeeze_3' and 'detection_output' layers but all these layers are supported by the CPU extension and GPU extension available in the openVINO toolkit. Thus, no custom layers were needed in order to use this model with the openVINO Inference Engine. Rather, we just need to add the right extension to the Inference engine core.

## Assess Model Use Cases

Some of the potential use cases of the People Counter App are:

1. Monitoring of civilian movements within public places such as parks, banks, theme parks, cinemas and so on during the period of the COVID-19 lockdown.

2. To detect intruders in restricted areas or private properties. The app does this by raising an alarm when it detects a person within the camera's range of vision.

3. Crime detection in city-wide areas through the use of already installed cameras within the city's perimeter.

4. Searching city-wide or nation-wide areas for wanted felons using cameras in a city combined with the people counter app equipped with a model trained to detect that individual felon or a group of felons.

5. For monitoring worker behaviors and mass-movement trends of workers in a factory, mine or production facility where man-power is utilized.

6. When combined with armed defense systems, it can be used to track, locate and deter intruders of highly restricted areas and locations.

### Inference time

When the pre-conversion model was tested on the classroom workspace, The minimum inference time of the pre-conversion model was 928.126ms and its maximum inference time was 942.889ms. This resulted an average inference time of 935.5075ms. After testing the pre-conversion model on a local device, the results were a lot different. The minimum inference time of the pre-conversion model reduced to 90.564ms and the maximum time of the model reduced to 94.254ms.

## Assess Effects on End User Needs

Lighting, model accuracy, weather, visibility and image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

- **Lighting:** The lighting in the camera's area of observation is the most important in order to obtain a good result. If the lighting in the image is poor or the area being monitored is dark, the model would not be able to detect any person within the video frame. The lighting can be affected by many factors. In the outdoors, lighting is primarily affected by the time of day, amount of artificial light sources (like street lamps) in the area, amount of lit sources of fires and animals. In the indoors, lighting is primarily sourced artificially. Some places also use glow-in-the-dark plants and animals and fires as indoor sources of light.

- **Model accuracy:** Deployed edge models must be able to make highly accurate detections. This is because these models results are usually required for real time usage and if they are deployed with models that have poor accuracy, it would make the app a lot less reliable. Thus the model must always only send results that have a high degree of accuracy to the app. Taking into account the amount of hinderances that come with deploying in remote areas, a minimum of 60% accuracy is recommended for models used in edge applications.

- **Weather:** The performance of the people counter app is directly dependent on the visibility and lighting in the camera's area of view. A people counter app deployed in the outdoors may not perform optimally in limited visibility. This also holds true for the case of the weather. This is because weather is one of the factors that affects visibility in a region.

- **Image size:** Image size determines the input size of the model. The image size depends on the resolution of the camera. A higher camera resolution results in a larger image file but also produces a more detailed image while a lesser camera resolution would result in a smaller image file with lesser details. A model is able to make more accurate classifications with a higher degree of accuracy if the resolution of the image is high but the trade-off is that this can also increase latency during video streaming and use up more storage space than we would want.

- **Visibility:** The visibility in the area of deployment directly affects the accuracy of models in the Edge application. If the app is deploys in areas with large amounts of particles in the air such as desserts, the visibility would be reduced on windy days and during sandstorms by sand particles carried by the wind. Snowstorms, hail, rain and hurricanes and gales also reduce the visibility in the area while bright and clear weather such as sunny or cloudy weather provide high levels of visibility. Other natural phenomena also affect the level of visibility in a cameras range of view.

## Conclusion

Upon testing the people counter app with videos and images of objects besides people, for example cars, dogs etc, it was discovered that the app did not draw any bounding boxes for images other than those of people. All other object classes were ignored except the 'person' class. A few videos and images have been provided in the `resource` folder located in the home directory to use to confirm this.

In conclusion, the app works as intended and identifies only people present in the video or image. It also performs all calculations and inferences as intended and returns the desired results in the desired formats through the UI accessed via the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) using a web browser or in the local storage at the `outputs\videos\` for video outputs and `outputs\images\` directory for images.
