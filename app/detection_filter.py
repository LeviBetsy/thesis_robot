def find_class_id(model, class_name):
    """
    Finds the class ID corresponding to a given class name from the model result.

    Args:
        model_result: The result object returned by the YOLO model inference.
        class_name: The name of the class for which to find the class ID.

    Returns:
        The class ID corresponding to the specified class name, or None if not found.
    """
    for idx, name in enumerate(model.names):
        if name.lower() == class_name.lower():
            return idx
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
    bounds = []
    for box in model_result[0].boxes:
        if box.cls == class_id:
            print(type(box.xyxy[0]))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bounds.append((x1, y1, x2, y2))
    return bounds