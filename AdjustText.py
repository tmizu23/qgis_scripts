##AdjustText=name
##Reset=boolean True

from qgis.core import *
from qgis.gui import *
from qgis.utils import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import itertools
from operator import itemgetter

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

def move_texts(bboxes):
    for bbox in bboxes:
        featureId = bbox['featureId']
        x=bbox['orgx']
        y=bbox['orgy']
        ha=bbox['ha']
        va=bbox['va']
        set_label_position(featureId,x,y,ha,va)

# def get_text_position(text,layer):
#     feature = getFeatureById(layer, text.featureId)
#     return (feature['x'],feature['y'])

def set_label_position(featureId,x,y,ha,va):
    projectCRSSrsid = canvas.mapSettings().destinationCrs().srsid()
    layerCRSSrsid = layer.crs().srsid()
    if projectCRSSrsid != layerCRSSrsid:
        crs_trans = QgsCoordinateTransform(projectCRSSrsid,layerCRSSrsid)
        p = crs_trans.transform(QgsPoint(x, y))
        x,y = p[0],p[1]

    feature = getFeatureById(layer, featureId)
    layer.startEditing()
    feature['label_x'] = round(x,10)
    feature['label_y'] = round(y,10)
    feature['label_ha'] = ha
    feature['label_va'] = va
    layer.updateFeature(feature)
    layer.commitChanges()
    #feature = getFeatureById(layer, text.featureId)
    #log("setx:{},sety:{}".format(x, y))

def get_point_position(text):
    feature = getFeatureById(layer, text.featureId)
    p = feature.geometry().asPoint()
    return (p[0],p[1])

def set_bboxes(bboxes,delta_x, delta_y):
    for i,(bbox,dx, dy) in enumerate(zip(bboxes,delta_x, delta_y)):
        bbox['orgx'] = bbox['orgx']+dx
        bbox['orgy'] = bbox['orgy']+dy
        bbox['xmin'] = bbox['xmin']+dx
        bbox['ymin'] = bbox['ymin']+dy
        bbox['xmax'] = bbox['xmax']+dx
        bbox['ymax'] = bbox['ymax']+dy
        bboxes[i]=bbox
    return bboxes

def set_bbox_align(bbox, ha,va):
    orgx = bbox['orgx']
    orgy = bbox['orgy']
    width = bbox['width']
    height = bbox['height']

    if ha=="left":
        bbox['xmin'] = orgx
        bbox['xmax'] = orgx + width
    elif ha=="right":
        bbox['xmin'] = orgx - width
        bbox['xmax'] = orgx
    elif ha == "center":
        bbox['xmin'] = orgx - width/2.0
        bbox['xmax'] = orgx + width/2.0

    if va=="bottom":
        bbox['ymin'] = orgy
        bbox['ymax'] = orgy + height
    elif va=="top":
        bbox['ymin'] = orgy - height
        bbox['ymax'] = orgy
    elif va == "center":
        bbox['ymin'] = orgy - height/2.0
        bbox['ymax'] = orgy + height/2.0

    bbox['ha'] = ha
    bbox['va'] = va

    return bbox

def get_bboxes(texts,expand):
    #ha:left,va:bottomの時のラベル情報を取得
    bboxes = [{'featureId': text.featureId,
               'orgx': text.cornerPoints[0][0] + text.width / 2.0,
               'orgy': text.cornerPoints[0][1] + text.height / 2.0,
               'xmin': text.cornerPoints[0][0] - text.width * (expand[0] - 1) / 2.0,
               'xmax': text.cornerPoints[2][0] + text.width * (expand[0] - 1) / 2.0,
               'ymin': text.cornerPoints[0][1] - text.height * (expand[1] - 1) / 2.0,
               'ymax': text.cornerPoints[2][1] + text.height * (expand[1] - 1) / 2.0,
               'width': text.width * expand[0], 'height': text.height * expand[1],
               'ha':"center",'va':"center"}
              for text in texts]
    # for bbox in bboxes:
    #     log("xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"],bbox["xmax"],bbox["ymin"],bbox["ymax"]))
    return bboxes

def get_midpoint(bbox):
    cx = (bbox["xmin"]+bbox["xmax"])/2
    cy = (bbox["ymin"]+bbox["ymax"])/2
    return cx, cy

def get_points_inside_bbox(x, y, bbox):
    """Return the indices of points inside the given bbox."""
    x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    #log("x1:{},x:{},x2:{},{}".format(x1,x,x2,x>x1))
    x_in = np.logical_and(x>x1, x<x2)
    y_in = np.logical_and(y>y1, y<y2)
    #log("xin:{},yin:{}".format(x_in,y_in))
    return np.asarray(np.nonzero(x_in & y_in)[0])

def overlap_bbox_and_point(bbox, xp, yp):
    """Given a bbox that contains a given point, return the (x, y) displacement
    necessary to make the bbox not overlap the point."""
    cx, cy = get_midpoint(bbox)

    dir_x = np.sign(cx-xp)
    dir_y = np.sign(cy-yp)

    if dir_x == -1:
        dx = xp - bbox['xmax']
    elif dir_x == 1:
        dx = xp - bbox['xmin']
    else:
        dx = 0

    if dir_y == -1:
        dy = yp - bbox['ymax']
    elif dir_y == 1:
        dy = yp - bbox['ymin']
    else:
        dy = 0
    return dx, dy

def intersection_size(bbox1, bbox2):
    """
    Return the intersection of the two bboxes or None
    if they do not intersect.
    """
    x0 = np.maximum(bbox1['xmin'], bbox2['xmin'])
    x1 = np.minimum(bbox1['xmax'], bbox2['xmax'])
    y0 = np.maximum(bbox1['ymin'], bbox2['ymin'])
    y1 = np.minimum(bbox1['ymax'], bbox2['ymax'])
    return (x1-x0,y1-y0) if x0 <= x1 and y0 <= y1 else None


def optimally_align_text(x,y,bboxes):
    ha = ['left', 'right', 'center']
    va = ['bottom', 'top', 'center']
    alignment = [(h,v) for h, v in itertools.product(ha,va)]
    for i, bbox in enumerate(bboxes):
        counts = []
        for h, v in alignment:
            bbox = set_bbox_align(bbox,h,v)
            #log("{},{}".format(h,v))
            #log("xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]))
            c = len(get_points_inside_bbox(x, y, bbox))
            intersections = [intersection_size(bbox, bbox2) if i!=j else None for j, bbox2 in enumerate(bboxes) ]
            intersections = sum([abs(b[0]*b[1]) if b is not None else 0 for b in intersections])
            counts.append((c, intersections))
        # Most important: prefer alignments that keep the text inside the axes.
        # If tied, take the alignments that minimize the number of x, y points
        # contained inside the text.
        # Break any remaining ties by minimizing the total area of intersections
        # with all text bboxes and other objects to avoid.
        a, value = min(enumerate(counts), key=itemgetter(1))
        bbox = set_bbox_align(bbox,alignment[a][0],alignment[a][1])
        bboxes[i] = bbox
    return bboxes

def repel_text(bboxes):
    xmins = [bbox['xmin'] for bbox in bboxes]
    xmaxs = [bbox['xmax'] for bbox in bboxes]
    ymaxs = [bbox['ymax'] for bbox in bboxes]
    ymins = [bbox['ymin'] for bbox in bboxes]
    #log("xmins:{}".format(xmins))
    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)

    for i, bbox1 in enumerate(bboxes):
        overlaps = get_points_inside_bbox(np.array(xmins*2+xmaxs*2), np.array((ymins+ymaxs)*2),bbox1) % len(bboxes)
        overlaps = np.unique(overlaps)
        #log("{}".format(overlaps))
        for j in overlaps:
            bbox2 = bboxes[j]
            x, y = intersection_size(bbox1, bbox2)
            overlaps_x[i, j] = x
            overlaps_y[i, j] = y
            overlap_directions_x[i, j] = np.sign(bbox1['xmin'] - bbox2['xmin'])
            overlap_directions_y[i, j] = np.sign(bbox1['ymin'] - bbox2['ymin'])

    move_x = overlaps_x*overlap_directions_x
    move_y = overlaps_y*overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(overlaps_x), np.sum(overlaps_y)
    return delta_x, delta_y, q

def repel_text_from_points(x, y, bboxes):
    move_x = np.zeros((len(bboxes), len(x)))
    move_y = np.zeros((len(bboxes), len(x)))
    for i, bbox in enumerate(bboxes):
        xy_in = get_points_inside_bbox(x, y, bbox)
        #log("xmin:{},x:{},xmax:{},xy_in:{}".format(bbox["xmin"],x,bbox["xmax"],xy_in))
        for j in xy_in:
            xp, yp = x[j], y[j]
            dx, dy = overlap_bbox_and_point(bbox, xp, yp)

            move_x[i, j] = dx
            move_y[i, j] = dy

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)
    q = np.sum(np.abs(move_x)), np.sum(np.abs(move_y))

    return delta_x, delta_y, q


# def repel_text_from_axes(texts, extent,layer):
#     bboxes = get_bboxes(texts)
#     log("{}".format(len(bboxes)))
#     xmin, xmax = extent.xMinimum(),extent.xMaximum()
#     ymin, ymax = extent.yMinimum(),extent.yMaximum()
#     log("{},{},{},{}".format(xmin, xmax, ymin, ymax))
#     for i, bbox in enumerate(bboxes):
#         x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
#         log("{},{},{},{}".format(x1, y1, x2, y2))
#         dx, dy = 0, 0
#         if x1 < xmin:
#             dx = xmin - x1
#         if x2 > xmax:
#             dx = xmax - x2
#         if y1 < ymin:
#             dy = ymin - y1
#         if y2 > ymax:
#             dy = ymax - y2
#         if dx or dy:
#             x, y = get_text_position(texts[i], layer)
#             newx, newy = x + dx, y + dy
#             log("{},{}".format(newx,newy))
#             set_text_position(texts[i].featureId,layer,newx,newy)


def reset_label_position(features):
    layer.startEditing()
    pr = layer.dataProvider()

    for col in ['label_x','label_y','label_ha','label_va']:
        idx=layer.fieldNameIndex(col)
        if idx!=-1:
            pr.deleteAttributes([layer.fieldNameIndex(col)])
            layer.updateFields()
    pr.addAttributes([QgsField('label_y', QVariant.Double, 'double', 20, 10),QgsField('label_x', QVariant.Double, 'double', 20, 10),QgsField('label_ha', QVariant.String),QgsField('label_va', QVariant.String)])
    # for col in ['label_x','label_y','label_ha','label_va']:
    #     idx=layer.fieldNameIndex(col)
    #     if idx==-1:
    #         if col in ['label_x','label_y']:
    #             pr.addAttributes([QgsField(col, QVariant.Double, 'double', 20, 4)])
    #         elif col in ['label_ha','label_va']:
    #             pr.addAttributes([QgsField(col, QVariant.String)])
    layer.updateFields()

    for feature in features:
        try:
            p = feature.geometry().asPoint()
            pr.changeAttributeValues({feature.id(): {pr.fieldNameMap()['label_x']: round(p[0], 10)}})
            pr.changeAttributeValues({feature.id(): {pr.fieldNameMap()['label_y']: round(p[1], 10)}})
            pr.changeAttributeValues({feature.id(): {pr.fieldNameMap()['label_ha']: "center"}})
            pr.changeAttributeValues({feature.id(): {pr.fieldNameMap()['label_va']: "center"}})
            layer.updateFeature(feature)
            feature["label_x"] = round(p[0], 10)
            feature["label_y"] = round(p[1], 10)
            feature["label_ha"] = "center"
            feature["label_va"] = "center"
            layer.updateFeature(feature)
        except:
            pass
    layer.commitChanges()


def adjust_text(lim=500,force_text=(0.1, 0.25), force_points=(0.2, 0.5),precision=0.01):
    if layer is not None:
        features = layer.selectedFeatures()
        if len(features) == 0:
            features = layer.getFeatures()
        lr = canvas.labelingResults()
        extent = canvas.extent()
        texts = lr.labelsWithinRect(extent)
        bboxes = get_bboxes(texts, expand=(1.05, 1.2))
        orig_xy = [get_point_position(text) for text in texts]
        orig_x = np.array([xy[0] for xy in orig_xy])
        orig_y = np.array([xy[1] for xy in orig_xy])
        log("len:{}".format(len(orig_xy)))
        x,y = orig_x,orig_y
        force_text = float_to_tuple(force_text)
        force_points = float_to_tuple(force_points)

        sum_width = np.sum(list(map(lambda bbox: bbox['width'], bboxes)))
        sum_height = np.sum(list(map(lambda bbox: bbox['height'], bboxes)))
        precision_x = precision * sum_width
        precision_y = precision * sum_height
        #log("w:{},h:{},pw:{},ph:{}".format(sum_width,sum_height,precision_x,precision_y))

        bboxes = optimally_align_text(x,y,bboxes)
        #extent = canvas.extent()
        #repel_text_from_axes(texts,extent,layer)
        history = [(np.inf, np.inf)]*10
        for i in xrange(lim):
            log("i:{}".format(i))
            d_x_text, d_y_text, q1 = repel_text(bboxes)
            #d_x_text, d_y_text, q1 = [0] * len(texts), [0] * len(texts), (0, 0)
            d_x_points, d_y_points, q2 = repel_text_from_points(x, y, bboxes)
            #d_x_points, d_y_points, q2 = [0] * len(texts), [0] * len(texts), (0, 0)
            #log("d_x:{},d_y{},q2{}".format(d_x_points, d_y_points, q2))
            #log("d_x:{},d_y{},q2{}".format(d_x_text, d_y_text, q1))

            dx = (np.array(d_x_text) * force_text[0] +
                  np.array(d_x_points) * force_points[0])
            dy = (np.array(d_y_text) * force_text[1] +
                  np.array(d_y_points) * force_points[1])
            log("dx:{},dy{}".format(dx, dy))
            qx = np.sum([q[0] for q in [q1, q2]])
            qy = np.sum([q[1] for q in [q1, q2]])
            #log("qx:{},qy{}".format(qx,qy))
            histm = np.max(np.array(history), axis=0)
            #log("histm:{}".format(histm))
            history.pop(0)
            history.append((qx, qy))
            #log("history:{}".format(history))
            #move_texts(texts, layer, dx, dy)
            bboxes=set_bboxes(bboxes, dx, dy)
            if (qx < precision_x and qy < precision_y) or np.all([qx, qy] >= histm):
                break
        #for bbox in bboxes:
        #    log("xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]))
        move_texts(bboxes)
    else:
        iface.messageBar().pushMessage("Warning", "No layer", level=QgsMessageBar.WARNING)

def reset_labelLayer():
    if layer is not None:
        features = layer.selectedFeatures()
        if len(features) == 0:
            features = layer.getFeatures()
        #位置とalignを設定
        reset_label_position(features)
        #レイヤのラベルプロパティの設定。値で定義された式を設定
        uri = QgsApplication.qgisSettingsDirPath()[:-1] +"processing/scripts/"+ "label.qml"
        layer.loadNamedStyle(uri)
        layer.setCustomProperty("labeling/isExpression", True)
        palyr = QgsPalLayerSettings()
        palyr.readFromLayer(layer)
        #palyr.setDataDefinedProperty(QgsPalLayerSettings.Hali, True, True, "case when \"x\" < $x  THEN 'right' ELSE 'left' END", "halign")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.PositionX, True, False, "", "label_x")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.PositionY, True, False, "", "label_y")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.Hali, True, False,"", "label_ha")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.Vali, True, False, "", "label_va")
        palyr.writeToLayer(layer)
        canvas.refresh()


canvas = iface.mapCanvas()
layer = iface.activeLayer()
if Reset:
    reset_labelLayer()
else:
    adjust_text()