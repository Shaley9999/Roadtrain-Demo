How to count objects
++++++++++++++++++++++++

In the main ``while True:`` loop::

    if FLAGS.count:
        # count objects found
        counted_classes = utils.count_objects(
            pred_bbox, by_class=True, allowed_classes=allowed_classes)
        
        # draw bounding boxes and output count 
        image = utils.draw_bbox(
            frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes)
    else:
        # just draw the bounding boxes
        image = utils.draw_bbox(
            frame, pred_bbox, FLAGS.info, allowed_classes=allowed_classes)



Here is ``utils.count_objects()``::

    # function to count objects, can return total classes or count per class
    def count_objects(data, by_class=False, allowed_classes=list(read_class_names(cfg.YOLO.CLASSES).values())):
        boxes, scores, classes, num_objects = data
        
        #create dictionary to hold count of objects
        counts = dict()

        # if by_class = True then count objects per class
        if by_class:
            class_names = read_class_names(cfg.YOLO.CLASSES)
            # loop through total number of objects found
            for i in range(num_objects):
                # grab class index and convert into corresponding class name
                class_index = int(classes[i])
                class_name = class_names[class_index]
                if class_name in allowed_classes:
                    counts[class_name] = counts.get(class_name, 0) + 1
                else:
                    continue
        # else count total objects found
        else:
            counts['total object'] = num_objects
        return counts


In the ``draw_bbox`` function::

    for i in range(num_boxes):

        # SKIPPED Code

        if counted_classes != None:
        height_ratio = int(image_h / 25) + 5
        offset = 15
        for key, value in counted_classes.items():
            if key == "bench" or key == "bus" or key == "wine glass" or key == "sandwich" or key == "toothbrush":
                cv2.putText(image, "{}es detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            elif key == "sheep" or key == "skis" or key == "scissors":
                cv2.putText(image, "{} detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            elif key == "knife":
                cv2.putText(image, "knives detected: {}".format(value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            elif key == "mouse":
                cv2.putText(image, "mice detected: {}".format(value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            elif key == "person":
                cv2.putText(image, "people detected: {}".format(value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            offset += height_ratio
    return image