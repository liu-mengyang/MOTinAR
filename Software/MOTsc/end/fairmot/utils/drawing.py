import colorsys
import cv2

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b

def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)

def rectangle(image, x, y, w, h, color, label=None, thickness=1):
    """Draw a rectangle.

    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.

    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 3, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 255, 255), thickness)

def draw_detections(img, dets):
    for det in dets:
        color=(0,0,0)
        rectangle(img, det.bbx[0], det.bbx[1], det.bbx[2]-det.bbx[0], det.bbx[3]-det.bbx[1], color)

def draw_trackers(img, tracks,thickness=1):
    for track in tracks:
        color = create_unique_color_uchar(track.track_id)
        rectangle(img, *track.track_bbox, color, label=str(track.track_id), thickness=thickness)
