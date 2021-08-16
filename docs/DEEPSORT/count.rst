How to count objects
++++++++++++++++++++++

In the main ``while True:`` loop::

    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)

    if FLAGS.count:
        if by_class:
            objs = dict(Counter(names))
            for key, value in objs.items():
                if key == "bench" or key == "bus" or key == "wine glass" or key == "sandwich" or key == "toothbrush":
                    cv2.putText(frame, "{}es detected: {}".format(
                        key, value), (indent, offset), font, font_size, red, font_thickness)
                    print("Number of {}es: {}".format(key, value))
                elif key == "sheep" or key == "skis" or key == "scissors":
                    cv2.putText(frame, "{} detected: {}".format(
                        key, value), (indent, offset), font, font_size, red, font_thickness)
                    print("Number of {}: {}".format(key, value))
                elif key == "knife":
                    cv2.putText(frame, "knives detected: {}".format(
                        value), (indent, offset), font, font_size, red, font_thickness)
                    print("Number of knives: ", value)
                elif key == "mouse":
                    cv2.putText(frame, "mice detected: {}".format(
                        value), (indent, offset), font, font_size, red, font_thickness)
                    print("Number of mice: ", value)
                elif key == "person":
                    cv2.putText(frame, "people detected: {}".format(
                        value), (indent, offset), font, font_size, blue, font_thickness)
                    print("Number of people: ", value)
                else:
                    print("Number of {}s: {}".format(key, value))
                    cv2.putText(frame, "{}s detected: {}".format(
                        key, value), (indent, offset), font, font_size, red, font_thickness)
                offset += 40
        else:
            cv2.putText(frame, "Objects being tracked: {}".format(
                count), (indent, offset), font, font_size, red, font_thickness)
            print("Objects being tracked: {}".format(count))