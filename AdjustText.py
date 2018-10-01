##AdjustText=name
##Lat=number 35.0

from qgis.core import *
from qgis.gui import *
from qgis.utils import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np

def log(msg):
    QgsMessageLog.logMessage(msg, 'MyPlugin', QgsMessageLog.INFO)

def getFeatureById(layer,featid):
    features = [f for f in layer.getFeatures(QgsFeatureRequest().setFilterFids([featid]))]
    if len(features) != 1:
        return None
    else:
        return features[0]

def float_to_tuple(a):
    try:
        a = float(a)
        return (a, a)
    except TypeError:
        assert len(a)==2
        try:
            b = float(a[0]), float(a[1])
        except TypeError:
            raise TypeError('Force values must be castable to floats')
        return b

def get_text_position(text,layer):
    feature = getFeatureById(layer, text.featureId)
    return (feature['x'],feature['y'])

def get_point_position(text,layer):
    feature = getFeatureById(layer, text.featureId)
    p = feature.geometry().asPoint()
    return (p[0],p[1])

def set_text_position(text,layer,x,y):
    feature = getFeatureById(layer, text.featureId)
    layer.startEditing()
    feature['x'] = round(x,4)
    feature['y'] = round(y,4)
    #log("setx:{},sety:{}".format(x, y))
    layer.updateFeature(feature)
    layer.commitChanges()

def get_bboxes(objs):
    return [{'xmin':lrl.cornerPoints[0][0],'xmax':lrl.cornerPoints[2][0],'ymin':lrl.cornerPoints[0][1],'ymax':lrl.cornerPoints[2][1],'width':lrl.width,'height':lrl.height} for lrl in objs]

def get_midpoint(bbox):
    cx = (bbox.x0+bbox.x1)/2
    cy = (bbox.y0+bbox.y1)/2
    return cx, cy

def get_points_inside_bbox(x, y, bbox):
    """Return the indices of points inside the given bbox."""
    x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    x_in = np.logical_and(x>x1, x<x2)
    y_in = np.logical_and(y>y1, y<y2)
    x_in
    return np.asarray(np.nonzero(x_in & y_in)[0])

def overlap_bbox_and_point(bbox, xp, yp):
    """Given a bbox that contains a given point, return the (x, y) displacement
    necessary to make the bbox not overlap the point."""
    cx, cy = get_midpoint(bbox)

    dir_x = np.sign(cx-xp)
    dir_y = np.sign(cy-yp)

    if dir_x == -1:
        dx = xp - bbox.xmax
    elif dir_x == 1:
        dx = xp - bbox.xmin
    else:
        dx = 0

    if dir_y == -1:
        dy = yp - bbox.ymax
    elif dir_y == 1:
        dy = yp - bbox.ymin
    else:
        dy = 0
    return dx, dy

def repel_text(texts,move=False):
    bboxes = get_bboxes(texts)
    xmins = [bbox['xmin'] for bbox in bboxes]
    xmaxs = [bbox['xmax'] for bbox in bboxes]
    ymaxs = [bbox['ymax'] for bbox in bboxes]
    ymins = [bbox['ymin'] for bbox in bboxes]

    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)

    for i, bbox1 in enumerate(bboxes):
        overlaps = get_points_inside_bbox(xmins*2+xmaxs*2, (ymins+ymaxs)*2,bbox1) % len(bboxes)
        overlaps = np.unique(overlaps)
        #log("{}".format(overlaps))
        for j in overlaps:
            bbox2 = bboxes[j]
            x, y = bbox1.intersection(bbox1, bbox2).size
            overlaps_x[i, j] = x
            overlaps_y[i, j] = y
            direction = np.sign(bbox1.extents - bbox2.extents)[:2]
            overlap_directions_x[i, j] = direction[0]
            overlap_directions_y[i, j] = direction[1]

    move_x = overlaps_x*overlap_directions_x
    move_y = overlaps_y*overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(overlaps_x), np.sum(overlaps_y)
    if move:
        pass
        #move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q

def repel_text_from_points(x, y, texts, move=False):

    bboxes = get_bboxes(texts)

    # move_x[i,j] is the x displacement of the i'th text caused by the j'th point
    move_x = np.zeros((len(bboxes), len(x)))
    move_y = np.zeros((len(bboxes), len(x)))
    for i, bbox in enumerate(bboxes):
        xy_in = get_points_inside_bbox(x, y, bbox)
        for j in xy_in:
            xp, yp = x[j], y[j]
            dx, dy = overlap_bbox_and_point(bbox, xp, yp)

            move_x[i, j] = dx
            move_y[i, j] = dy

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)
    q = np.sum(np.abs(move_x)), np.sum(np.abs(move_y))
    # if move:
    #     move_texts(texts, delta_x, delta_y, bboxes, ax=ax)
    return delta_x, delta_y, q


def repel_text_from_axes(texts, extent,layer):
    bboxes = get_bboxes(texts)
    log("{}".format(len(bboxes)))
    xmin, xmax = extent.xMinimum(),extent.xMaximum()
    ymin, ymax = extent.yMinimum(),extent.yMaximum()
    log("{},{},{},{}".format(xmin, xmax, ymin, ymax))
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
        log("{},{},{},{}".format(x1, y1, x2, y2))
        dx, dy = 0, 0
        if x1 < xmin:
            dx = xmin - x1
        if x2 > xmax:
            dx = xmax - x2
        if y1 < ymin:
            dy = ymin - y1
        if y2 > ymax:
            dy = ymax - y2
        if dx or dy:
            x, y = get_text_position(texts[i], layer)
            newx, newy = x + dx, y + dy
            log("{},{}".format(newx,newy))
            set_text_position(texts[i],layer,newx,newy)

def move_texts(texts, layer,delta_x, delta_y):
    for text,dx, dy in zip(texts,delta_x, delta_y):
        x, y = get_text_position(text,layer)
        newx = x + dx
        newy = y + dy
        set_text_position(text, layer, newx, newy)

def adjust_text(lim=500,force_text=(0.1, 0.25), force_points=(0.2, 0.5),precision=0.01):
    canvas=iface.mapCanvas()
    layer = iface.activeLayer()

    if layer is not None:
        features = layer.selectedFeatures()
        if len(features) == 0:
            features = layer.getFeatures()

        lr = canvas.labelingResults()
        texts = lr.labelsWithinRect(QgsRectangle(-1000000,-1000000,1000000,1000000))
        orig_xy = [get_point_position(text,layer) for text in texts]
        orig_x = [xy[0] for xy in orig_xy]
        orig_y = [xy[1] for xy in orig_xy]
        #log("len:{}".format(len(orig_xy)))
        for text,x,y in zip(texts,orig_x,orig_y):
            set_text_position(text,layer,x,y)
        x,y = orig_x,orig_y
        force_text = float_to_tuple(force_text)
        force_points = float_to_tuple(force_points)
        bboxes = get_bboxes(texts)
        sum_width = np.sum(list(map(lambda bbox: bbox['width'], bboxes)))
        sum_height = np.sum(list(map(lambda bbox: bbox['height'], bboxes)))
        precision_x = precision * sum_width
        precision_y = precision * sum_height


        #extent = canvas.extent()
        #repel_text_from_axes(texts,extent,layer)

        history = [(np.inf, np.inf)]*10
        for i in xrange(lim):
        #   d_x_text, d_y_text, q1 = repel_text(texts)
            d_x_text, d_y_text, q1 = [0] * len(texts), [0] * len(texts), (0, 0)
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, texts)
            log("{},{},{}".format(d_x_points, d_y_points, q2))

            dx = (np.array(d_x_text) * force_text[0] +
                  np.array(d_x_points) * force_points[0])
            dy = (np.array(d_y_text) * force_text[1] +
                  np.array(d_y_points) * force_points[1])
            qx = np.sum([q[0] for q in [q1, q2]])
            qy = np.sum([q[1] for q in [q1, q2]])
            histm = np.max(np.array(history), axis=0)
            history.pop(0)
            history.append((qx, qy))
            move_texts(texts, layer, dx, dy)
            if (qx < precision_x and qy < precision_y) or np.all([qx, qy] >= histm):
                break
    else:
        iface.messageBar().pushMessage("Warning", "No layer", level=QgsMessageBar.WARNING)

adjust_text()

# log("{}".format(extent))
# for lrl in lr.labelsWithinRect(extent):
#     if lrl.layerID == layer.id():
#         angle = 90 - QgsPoint(lrl.cornerPoints[0]).azimuth(QgsPoint(lrl.cornerPoints[1]))
#         xmin = lrl.cornerPoints[0][0]
#         xmax = lrl.cornerPoints[2][0]
#         ymin = lrl.cornerPoints[0][1]
#         ymax = lrl.cornerPoints[2][1]
#         rlabel = angle
#         featid = lrl.featureId
#         feature=getFeatureById(layer,featid)
#         p = feature.geometry().asPoint()
#         log("{},{},{},{},{},{},{},{}".format(featid,p[0],p[1],xmin,xmax,ymin,ymax,angle))

# def move_texts():
#     layer.startEditing()
#     for feature in features:
#         p = feature.geometry().asPoint()
#         x = feature['x']
#         feature['x'] = x+100
#         layer.updateFeature(feature)
#         log("{}".format(x))
#     layer.commitChanges()