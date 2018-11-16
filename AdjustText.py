##AdjustText=name
##Reset=boolean True

from qgis.core import *
from qgis.gui import *
from qgis.utils import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np
import itertools
import math
from operator import itemgetter
import sys


def log(msg):
    QgsMessageLog.logMessage(msg, 'MyPlugin', QgsMessageLog.INFO)


def getFeatureById(layer, featid):
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
        assert len(a) == 2
        try:
            b = float(a[0]), float(a[1])
        except TypeError:
            raise TypeError('Force values must be castable to floats')
        return b


def move_texts(bboxes):
    for bbox in bboxes:
        featureId = bbox['featureId']
        x = bbox['orgx']
        y = bbox['orgy']
        ha = bbox['ha']
        va = bbox['va']
        set_label_position(featureId, x, y, ha, va)


# def get_text_position(text,layer):
#     feature = getFeatureById(layer, text.featureId)
#     return (feature['x'],feature['y'])

def set_label_position(featureId, x, y, ha, va):
    projectCRSSrsid = canvas.mapSettings().destinationCrs().srsid()
    layerCRSSrsid = layer.crs().srsid()
    if projectCRSSrsid != layerCRSSrsid:
        crs_trans = QgsCoordinateTransform(projectCRSSrsid, layerCRSSrsid)
        p = crs_trans.transform(QgsPoint(x, y))
        x, y = p[0], p[1]

    feature = getFeatureById(layer, featureId)
    layer.startEditing()
    feature['label_x'] = round(x, 10)
    feature['label_y'] = round(y, 10)
    feature['label_ha'] = ha
    feature['label_va'] = va
    layer.updateFeature(feature)
    layer.commitChanges()
    # feature = getFeatureById(layer, text.featureId)
    # log("setx:{},sety:{}".format(x, y))


def get_point_position(text):
    # データポイントの位置を返す
    feature = getFeatureById(layer, text.featureId)
    p = feature.geometry().asPoint()
    return (p[0], p[1])


def set_bboxes(bboxes, delta_x, delta_y):
    for i, (bbox, dx, dy) in enumerate(zip(bboxes, delta_x, delta_y)):
        bbox['orgx'] = bbox['orgx'] + dx
        bbox['orgy'] = bbox['orgy'] + dy
        bbox['xmin'] = bbox['xmin'] + dx
        bbox['ymin'] = bbox['ymin'] + dy
        bbox['xmax'] = bbox['xmax'] + dx
        bbox['ymax'] = bbox['ymax'] + dy
        bboxes[i] = bbox
    return bboxes


def set_bbox_align(bbox, ha, va):
    orgx = bbox['orgx']
    orgy = bbox['orgy']
    width = bbox['width']
    height = bbox['height']

    if ha == "left":
        bbox['xmin'] = orgx
        bbox['xmax'] = orgx + width
    elif ha == "right":
        bbox['xmin'] = orgx - width
        bbox['xmax'] = orgx
    elif ha == "center":
        bbox['xmin'] = orgx - width / 2.0
        bbox['xmax'] = orgx + width / 2.0

    if va == "bottom":
        bbox['ymin'] = orgy
        bbox['ymax'] = orgy + height
    elif va == "top":
        bbox['ymin'] = orgy - height
        bbox['ymax'] = orgy
    elif va == "center":
        bbox['ymin'] = orgy - height / 2.0
        bbox['ymax'] = orgy + height / 2.0

    bbox['ha'] = ha
    bbox['va'] = va

    return bbox


def get_bboxes(texts, expand, random):
    # ha:left,va:bottomの時のラベル情報を取得
    bboxes = [{'featureId': text.featureId,
               'orgx': text.cornerPoints[0][0] + text.width / 2.0 + r,
               'orgy': text.cornerPoints[0][1] + text.height / 2.0 + r,
               'xmin': text.cornerPoints[0][0] - text.width * (expand[0] - 1) / 2.0 + r,
               'xmax': text.cornerPoints[2][0] + text.width * (expand[0] - 1) / 2.0 + r,
               'ymin': text.cornerPoints[0][1] - text.height * (expand[1] - 1) / 2.0 + r,
               'ymax': text.cornerPoints[2][1] + text.height * (expand[1] - 1) / 2.0 + r,
               'width': text.width * expand[0], 'height': text.height * expand[1],
               'ha': "center", 'va': "center"}
              for text, r in zip(texts, random)]
    # for bbox in bboxes:
    #     log("xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"],bbox["xmax"],bbox["ymin"],bbox["ymax"]))
    return bboxes


def get_dboxes(orig_xy, padding):
    # ポイントのpadding付きの範囲を取得
    dboxes = [{'xmin': xy[0] - padding[0],
               'xmax': xy[0] + padding[0],
               'ymin': xy[1] - padding[1],
               'ymax': xy[1] + padding[1],
               }
              for xy in orig_xy]
    return dboxes


def get_midpoint(bbox):
    cx = (bbox["xmin"] + bbox["xmax"]) / 2
    cy = (bbox["ymin"] + bbox["ymax"]) / 2
    return np.array([cx, cy])


def get_points_inside_bbox(x, y, bbox):
    """Return the indices of points inside the given bbox."""
    x1, y1, x2, y2 = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
    # log("x1:{},x:{},x2:{},{}".format(x1,x,x2,x>x1))
    x_in = np.logical_and(x > x1, x < x2)
    y_in = np.logical_and(y > y1, y < y2)
    # log("xin:{},yin:{}".format(x_in,y_in))
    return np.asarray(np.nonzero(x_in & y_in)[0])


def overlap_bbox_and_point(bbox, xp, yp):
    """Given a bbox that contains a given point, return the (x, y) displacement
    necessary to make the bbox not overlap the point."""
    cx, cy = get_midpoint(bbox)

    dir_x = np.sign(cx - xp)
    dir_y = np.sign(cy - yp)

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
    return (x1 - x0, y1 - y0) if x0 <= x1 and y0 <= y1 else None


def optimally_align_text(x, y, bboxes):
    ha = ['left', 'right', 'center']
    va = ['bottom', 'top', 'center']
    alignment = [(h, v) for h, v in itertools.product(ha, va)]
    for i, bbox in enumerate(bboxes):
        counts = []
        for h, v in alignment:
            bbox = set_bbox_align(bbox, h, v)
            # log("{},{}".format(h,v))
            # log("xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]))
            c = len(get_points_inside_bbox(x, y, bbox))
            intersections = [intersection_size(bbox, bbox2) if i != j else None for j, bbox2 in enumerate(bboxes)]
            intersections = sum([abs(b[0] * b[1]) if b is not None else 0 for b in intersections])
            counts.append((c, intersections))
        # Most important: prefer alignments that keep the text inside the axes.
        # If tied, take the alignments that minimize the number of x, y points
        # contained inside the text.
        # Break any remaining ties by minimizing the total area of intersections
        # with all text bboxes and other objects to avoid.
        a, value = min(enumerate(counts), key=itemgetter(1))
        bbox = set_bbox_align(bbox, alignment[a][0], alignment[a][1])
        bboxes[i] = bbox
    return bboxes


def repel_text(bboxes):
    xmins = [bbox['xmin'] for bbox in bboxes]
    xmaxs = [bbox['xmax'] for bbox in bboxes]
    ymaxs = [bbox['ymax'] for bbox in bboxes]
    ymins = [bbox['ymin'] for bbox in bboxes]
    # log("xmins:{}".format(xmins))
    overlaps_x = np.zeros((len(bboxes), len(bboxes)))
    overlaps_y = np.zeros_like(overlaps_x)
    overlap_directions_x = np.zeros_like(overlaps_x)
    overlap_directions_y = np.zeros_like(overlaps_y)

    for i, bbox1 in enumerate(bboxes):
        overlaps = get_points_inside_bbox(np.array(xmins * 2 + xmaxs * 2), np.array((ymins + ymaxs) * 2), bbox1) % len(
            bboxes)
        overlaps = np.unique(overlaps)
        # log("{}".format(overlaps))
        for j in overlaps:
            bbox2 = bboxes[j]
            x, y = intersection_size(bbox1, bbox2)
            overlaps_x[i, j] = x
            overlaps_y[i, j] = y
            overlap_directions_x[i, j] = np.sign(bbox1['xmin'] - bbox2['xmin'])
            overlap_directions_y[i, j] = np.sign(bbox1['ymin'] - bbox2['ymin'])

    move_x = overlaps_x * overlap_directions_x
    move_y = overlaps_y * overlap_directions_y

    delta_x = move_x.sum(axis=1)
    delta_y = move_y.sum(axis=1)

    q = np.sum(overlaps_x), np.sum(overlaps_y)
    return delta_x, delta_y, q


def repel_text_from_points(x, y, bboxes):
    move_x = np.zeros((len(bboxes), len(x)))
    move_y = np.zeros((len(bboxes), len(x)))
    for i, bbox in enumerate(bboxes):
        xy_in = get_points_inside_bbox(x, y, bbox)
        # log("xmin:{},x:{},xmax:{},xy_in:{}".format(bbox["xmin"],x,bbox["xmax"],xy_in))
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

    for col in ['label_x', 'label_y', 'label_ha', 'label_va']:
        idx = layer.fieldNameIndex(col)
        if idx != -1:
            pr.deleteAttributes([layer.fieldNameIndex(col)])
            layer.updateFields()
    pr.addAttributes(
        [QgsField('label_y', QVariant.Double, 'double', 20, 10), QgsField('label_x', QVariant.Double, 'double', 20, 10),
         QgsField('label_ha', QVariant.String), QgsField('label_va', QVariant.String)])
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


def euclid(a, b):
    dist = a - b
    return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1])


def euclid2(a, b):
    dist = a - b
    return dist[0] * dist[0] + dist[1] * dist[1]


def approximately_equal(x1, x2):
    return abs(x2 - x1) < sys.float_info.epsilon * 100


def line_intersect(p1, q1, p2, q2):
    # Special exception, where q1 and q2 are equal (do intersect)
    if q1[0] == q2[0] and q1[1] == q2[1]:
        return False
    # If line is point
    if p1[0] == q1[0] and p1[1] == q1[1]:
        return False
    if p2[0] == q2[0] and p2[1] == q2[1]:
        return False

    dy1 = q1[1] - p1[1]
    dx1 = q1[0] - p1[0]

    slope1 = dy1 / dx1
    intercept1 = q1[1] - q1[0] * slope1

    dy2 = q2[1] - p2[1]
    dx2 = q2[0] - p2[0]

    slope2 = dy2 / dx2
    intercept2 = q2[1] - q2[0] * slope2

    # check if lines vertical
    if approximately_equal(dx1, 0.0):
        if approximately_equal(dx2, 0.0):
            return False
        else:
            x = p1[0]
            y = slope2 * x + intercept2
    elif approximately_equal(dx2, 0.0):
        x = p2[0]
        y = slope1 * x + intercept1
    else:
        if approximately_equal(slope1, slope2):
            return False
        x = (intercept2 - intercept1) / (slope1 - slope2)
        y = slope1 * x + intercept1

    if x < p1[0] and x < q1[0]:
        return False
    elif x > p1[0] and x > q1[0]:
        return False
    elif y < p1[1] and y < q1[1]:
        return False
    elif y > p1[1] and y > q1[1]:
        return False
    elif x < p2[0] and x < q2[0]:
        return False
    elif x > p2[0] and x > q2[0]:
        return False
    elif y < p2[1] and y < q2[1]:
        return False
    elif y > p2[1] and y > q2[1]:
        return False
    else:
        return True

def set_bbox(bbox,velocities):
    dx, dy = velocities[0],velocities[1]
    bbox['orgx'] = bbox['orgx']+dx
    bbox['orgy'] = bbox['orgy']+dy
    bbox['xmin'] = bbox['xmin']+dx
    bbox['ymin'] = bbox['ymin']+dy
    bbox['xmax'] = bbox['xmax']+dx
    bbox['ymax'] = bbox['ymax']+dy
    return bbox

def put_within_bounds(b, xlim, ylim, force=1e-5):
    width = math.fabs(b["xmin"] - b["xmax"])
    height = math.fabs(b["ymin"] - b["ymax"])
    if b["xmin"] < xlim[0]:
        b["xmin"] = xlim[0]
        b["xmax"] = b["xmin"] + width
    elif b["xmax"] > xlim[1]:
        b["xmax"] = xlim[1]
        b["xmin"] = b["xmax"] - width

    if b["ymin"] < ylim[0]:
        b["ymin"] = ylim[0]
        b["ymax"] = b["ymin"] + height
    elif b["ymax"] > ylim[1]:
        b["ymax"] = ylim[1]
        b["ymin"] = b["ymax"] - height

    return b


def spring_force(a, b, force=0.000001):
    v = (a - b)
    f = force * v
    return f


def repel_force(a, b, force=0.000001):
    dx = math.fabs(a[0] - b[0])
    dy = math.fabs(a[1] - b[1])
    # Constrain the minimum distance, so it is never 0.
    d2 = max(dx * dx + dy * dy, 0.0004)
    # Compute a unit vector in the direction of the force.
    v = (a - b) / math.sqrt(d2)
    # Divide the force by the squared distance.
    f = force * v / d2
    if dx > dy:
        f[1] = f[1] * 2
    else:
        f[0] = f[0] * 2
    return f



def adjust_text(force_push=1e-6, force_pull=1e-2, maxiter=2000):
    if layer is None:
        iface.messageBar().pushMessage("Warning", "No layer", level=QgsMessageBar.WARNING)
        return

    # text情報
    features = layer.selectedFeatures()
    if len(features) == 0:
        features = layer.getFeatures()
    lr = canvas.labelingResults()
    extent = canvas.extent()
    texts = lr.labelsWithinRect(extent)

    n_texts = len(texts)
    r = np.random.normal(0, force_push, n_texts)
    bboxes = get_bboxes(texts, expand=(1.05, 1.2), random=r)
    xbounds = [0, 1]
    ybounds = [0, 1]

    orig_xy = np.array([get_point_position(text) for text in texts])
    dboxes = get_dboxes(orig_xy, padding=(0, 0))
    n_points = len(orig_xy)

    log("len:{}".format(len(orig_xy)))
    velocities = np.array([[0,0]]*n_texts)
    velocity_decay = 0.7
    iter = 0
    any_overlaps = True
    i_overlaps = True

    while (any_overlaps and iter < maxiter):
        iter = iter + 1
        any_overlaps = False
        # The forces get weaker over time.
        force_push = force_push * 0.99999
        force_pull = force_pull * 0.9999
        for i in range(n_texts):
            i_overlaps = False
            f = np.array([0, 0])
            ci = get_midpoint(bboxes[i])
            for j in range(n_points):
                if i == j:
                    # Repel the box from its data point.
                    if intersection_size(dboxes[i], bboxes[i]) is not None:
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, orig_xy[i], force_push)
                else:
                    cj = get_midpoint(bboxes[j])
                    # Repel the box from overlapping boxes.
                    if j < n_texts and intersection_size(bboxes[i], bboxes[j]) is not None:
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, cj, force_push)
                    # Repel the box from other data points.
                    if intersection_size(dboxes[j], bboxes[i]) is not None:
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, orig_xy[j], force_push)
            # Pull the box toward its original position.
            if not i_overlaps:
                f = f + spring_force(orig_xy[i], ci, force_pull)
            velocities[i] = velocities[i] * velocity_decay + f
            bboxes[i] = set_bbox(bboxes[i], velocities[i])
            # Put boxes within bounds
            #bboxes[i] = put_within_bounds(bboxes[i], xbounds, ybounds)
            # look for line clashes
            if not any_overlaps or iter % 5 == 0:
                for j in range(n_points):
                    cj = get_midpoint(bboxes[j])
                    ci = get_midpoint(bboxes[i])
                    # Switch label positions if lines overlap
                    if i != j and j < n_texts and line_intersect(ci, orig_xy[i], cj, orig_xy[j]):
                        any_overlaps = True
                        bboxes[i] = set_bbox(bboxes[i],spring_force(cj, ci, 1))
                        bboxes[j] = set_bbox(bboxes[j],spring_force(ci, cj, 1))
                        # Check if resolved
                        ci = get_midpoint(bboxes[i])
                        cj = get_midpoint(bboxes[j])
                        if line_intersect(ci, orig_xy[i], cj, orig_xy[j]):
                            bboxes[i] = set_bbox(bboxes[i],spring_force(cj, ci, 1.25))
                            bboxes[j] = set_bbox(bboxes[j],spring_force(ci, cj, 1.25))
    move_texts(bboxes)


def reset_labelLayer():
    if layer is not None:
        features = layer.selectedFeatures()
        if len(features) == 0:
            features = layer.getFeatures()
        # 位置とalignを設定
        reset_label_position(features)
        # レイヤのラベルプロパティの設定。値で定義された式を設定
        uri = QgsApplication.qgisSettingsDirPath()[:-1] + "processing/scripts/" + "label.qml"
        layer.loadNamedStyle(uri)
        layer.setCustomProperty("labeling/isExpression", True)
        palyr = QgsPalLayerSettings()
        palyr.readFromLayer(layer)
        # palyr.setDataDefinedProperty(QgsPalLayerSettings.Hali, True, True, "case when \"x\" < $x  THEN 'right' ELSE 'left' END", "halign")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.PositionX, True, False, "", "label_x")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.PositionY, True, False, "", "label_y")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.Hali, True, False, "", "label_ha")
        palyr.setDataDefinedProperty(QgsPalLayerSettings.Vali, True, False, "", "label_va")
        palyr.writeToLayer(layer)
        canvas.refresh()


canvas = iface.mapCanvas()
layer = iface.activeLayer()
if Reset:
    reset_labelLayer()
else:
    adjust_text()