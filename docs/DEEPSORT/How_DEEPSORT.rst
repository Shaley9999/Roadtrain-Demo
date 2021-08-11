How does it Work?
++++++++++++++++++

DeepSORT is a real time object **tracker**
It is an extension to SORT (Simple Online Realtime Tracker) by adding another distance metric.
The algorithm works in two parts:
- Detection: All objects are detected in the frame.
- Association: By looking at distance and predicted velocity a matching is performed for similar detections with respect to the previous frame. 

This allows objects to be tracked and given a particular ID. 

Deep SORT tracks the objects as opposed to YOLO which also results in smoother object detection
 as YOLO recalculates all boxes every frame of the video and doesn’t know if an object is the same one or different from the previous frame.

Subhead
==============


Subhead2
=========

1
------------

