def find_class_id(model, class_name):
    """
    Finds the class ID corresponding to a given class name from the model result.

    Args:
        model_result: The result object returned by the YOLO model inference.
        class_name: The name of the class for which to find the class ID.

    Returns:
        The class ID corresponding to the specified class name, or None if not found.
    """
    for class_id, name in model.names.items():
        if name.lower() == class_name.lower():
            return class_id
    return None

def get_bounds(model_result, class_id):
    """
    Extracts the bounding box coordinates for a specific class ID from the model result.

    Args:
        model_result: The result object returned by the YOLO model inference.
        class_id: The class ID for which to extract the bounding box.

    Returns:
        A list of bounding box coordinates (x1, y1, x2, y2) for the specified class ID.
    """
    boxes = model_result[0].boxes
    bounds = []
    for i in range(0, len(boxes.cls)):
        if boxes.cls[i] == class_id:
            xyxy = boxes.xyxy.data[i]
            bounds.append(xyxy)
            # bounds.append(boxes.xyxy)
            # print("i")
    return bounds