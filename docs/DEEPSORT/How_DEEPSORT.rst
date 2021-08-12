How does it Work?
++++++++++++++++++

DeepSORT vs YOLO
=================
DeepSORT is a real time object **tracker**
It is an extension to SORT (Simple Online Realtime Tracker) by adding another distance metric.
The algorithm works in **two** parts:
- **Detection:** All objects are detected in the frame.
- **Association:** By looking at distance and predicted velocity a matching is performed for similar detections with respect to the previous frame. 

This allows objects to be **tracked** and given a particular ID. 

Deep SORT tracks the objects as opposed to YOLO. This results in smoother object detection
as YOLO recalculates all boxes every frame of the video and so doesn’t know if an object is the same
one or different from the previous frame.

Object Tracking
================

In order to track objects from frame to frame in a video
methods such as `mean shift <https://en.wikipedia.org/wiki/Mean_shift>`_
and `optical flow <https://en.wikipedia.org/wiki/Optical_flow>`_ can be used.

However mean shift fails if objects are moving too fast and
optical flow is computationally complex and is prone to noise.
Both also can struggle if there is occlusion of the object being tracked.

Kalman Filter
-------------

A technique that solves these issues and is used widely is the
`Kalman Filter. <https://en.wikipedia.org/wiki/Kalman_filter>`_
A 8-dimensional state space holds the position of the bounding box along with its velocity.
(x, y, a, h, vx, vy, va, vh)

In order to track objects you can't see (or detect) you can assume a constant
velocity model and gaussian distribution in order to guestimate where the object
is based on the model of its motion.

When the object is able to be tracked, you rely more on the sensor data and thus
put more weight on it. The more the object is occluded the more weight you place
on the motion of the object.

The Kalman Filter is recursive and so more accurately tracks objects which are
changing speeds.

SORT (Simple Online Realtime Tracker)
--------------------------------------

SORT is composed of **4** core components:

- Detection
- Estimation
- Association
- Track Identity creation and destruction


Detection
~~~~~~~~~~~~
How detections can be made is explained in :Doc:`Detecting Objects </YOLOv4/How_YOLO>` . 

Estimation
~~~~~~~~~~~
After detection the detected bounding box is used to create an estimation of where the object will be
in the next frame via the Kalman Filter and linear assignment.

Association
~~~~~~~~~~~~
In assigning detections to existing targets, each target's bounding box geometry is estimated by predicting its
new location in the latest frame. The assignment cost matrix is then computed as the IOU distance between
each detection and all predicted boxes from the existing targets.

Below is iou_matching.py.
Note: linear_assignment.INFTY_COST = 100000::

    def iou(bbox, candidates):
        """Computer intersection over union.

        Parameters
        ----------
        bbox : ndarray
            A bounding box in format `(top left x, top left y, width, height)`.
        candidates : ndarray
            A matrix of candidate bounding boxes (one per row) in the same format
            as `bbox`.

        Returns
        -------
        ndarray
            The intersection over union in [0, 1] between the `bbox` and each
            candidate. A higher score means a larger fraction of the `bbox` is
            occluded by the candidate.

        """
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        return area_intersection / (area_bbox + area_candidates - area_intersection)


    def iou_cost(tracks, detections, track_indices=None,
                detection_indices=None):
        """An intersection over union distance metric.

        Parameters
        ----------
        tracks : List[deep_sort.track.Track]
            A list of tracks.
        detections : List[deep_sort.detection.Detection]
            A list of detections.
        track_indices : Optional[List[int]]
            A list of indices to tracks that should be matched. Defaults to
            all `tracks`.
        detection_indices : Optional[List[int]]
            A list of indices to detections that should be matched. Defaults
            to all `detections`.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape
            len(track_indices), len(detection_indices) where entry (i, j) is
            `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

        """
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            if tracks[track_idx].time_since_update > 1:
                cost_matrix[row, :] = linear_assignment.INFTY_COST
                continue

            bbox = tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            cost_matrix[row, :] = 1. - iou(bbox, candidates)
        return cost_matrix