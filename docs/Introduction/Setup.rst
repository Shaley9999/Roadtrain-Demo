Setup
++++++++++++

**Using Conda (What I used):**

Tensorflow CPU::

    conda env create -f conda-cpu.yml
    conda activate yolov4-cpu

Tensorflow GPU::

    conda env create -f conda-gpu.yml
    conda activate yolov4-gpu

**Using Pip:**

TensorFlow CPU::

    pip install -r requirements.txt

TensorFlow GPU::

    pip install -r requirements-gpu.txt

Downloading Pre-trained Weights
=================================
YOLOv4 comes with some pre-trained weights to detect the **80** classes
in the coco dataset. Download the weights
`here. <https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT>`_ 

Save this in the ``data`` folder in the repository.

Using Custom Trained YOLOv4 Weights
=====================================
Copy and paste your custom ``.weights`` file into the ``data`` folder
and copy and paste your custom ``.names`` into the ``data/classes/`` folder.

The only change within the code you need to make in order for your custom model to work
is on line **13** of ``core/config.py`` file.
Update the code to point at your custom ``.names`` file.

YOLOv4 and DEEPSORT using TensorFlow (.pb model)
==================================================
To implement YOLOv4 using TensorFlow,
first we convert the ``.weights`` into the corresponding TensorFlow model files
and then run the model::


    # Convert darknet weights to tensorflow
    python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 

    # Run yolov4 on images
    python detect.py --images ./data/images/kite.jpg

    # Run yolov4 on video
    python detect_video.py --video ./data/video/video.mp4

    # Run yolov4 on webcam
    python detect_video.py --video 0 

    # Run DeepSort on video
    python object_tracker.py --video ./data/video/video.mp4

    # Run DeepSort on webcam
    python object_tracker.py --video 0

.. note::
    You can also run the detector on multiple images at once by changing the
    ``--images`` flag like such ``--images "./data/images/kite.jpg, ./data/images/dog.jpg"``

Additional Commands
-------------------
For save_model:

- --weights: path to weights file (default: './data/yolov4.weights')
- --output: path to output (default: './checkpoints/yolov4-416')
- --input_size: define input size of export model (default: 416)
- --model: yolov3 or yolov4 (default: yolov4)

For detect.py and detect_video.py:

- --video: path to input video (use 0 for webcam) (default: './data/video/video.mp4')
- --images: path to input images as a string with images separated by "," (default: './data/images/kite.jpg')
- --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi) (default: None)
- --output_format: codec used in VideoWriter when saving video to file (default: 'XVID)
- --weights: path to weights file (default: './checkpoints/yolov4-416')
- --model: yolov3 or yolov4(default: yolov4)
- --count: count objects within video/images (default: False)
- --info: print info on detections to command line (default: False)
- --dont_show: dont show video/image output (default: False)
- --iou: iou threshold (default: 0.45)
- --size: resize images to (default: 416)
- --score: confidence threshold (default: 0.25)
  