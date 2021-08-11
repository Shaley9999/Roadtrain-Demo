
How does it Work?
++++++++++++

Classification vs Regression-based
===================================
There are two main ways of going about object detection:

#. 
    Classification based algorithms: There are mainly two stages in classification based algorithms.
    
        In the first stage, it will select a bunch of Region of Interest (ROI) in the image
        where the chances of objects are high.

        In the second stage, it will apply a Convolution Neural Network to these regions
        to detect the presence of an object.
    One of the problems with this method is, we have to execute the detector in each of the ROI,
    and that makes is slow and computationally expensive. One example of this type of algorithm is R-CNN.

#. 
    Regression-based algorithms: In this algorithm, there is no selection of interesting ROI in the image,
    instead of that, it will predict the classes and bounding boxes for the entire image at once.
    This makes detection faster than classification algorithms.

    YOLO (“You Only Look Once“) is an example of a regression-based algorithm.
    The YOLO detector is very fast so it is a great choice for self-driving cars and other applications
    where real-time object detection is required.

Detecting Objects
==================

The YOLO detector predicts the class of an object, its bounding boxes
and the probability of the class of object in the bounding box.
Each bounding box has the following parameters

- The **Centre position** of the bounding box in the image
- The **width** of the box 
- The **height** of the box 
- The **class** of object

Each bounding box is aasociated with a confidence score,
it is the probability of a class of object in that bounding box.

The image (or frame of video) is split into a grid of many cells.
The cell in which the centre of an object resides, is the cell responsible
for detecting that object.

The confidence represents the IOU between the predicted box and the actual box
(the ground truth box). 

IOU stands for Intersection Over Union and is the area of the intersection
of the predicted and ground truth boxes divided by the area of
the union of the same predicted and ground truth boxes.

.. image::
    /Images/IoU.png


1
------------

2
----------

