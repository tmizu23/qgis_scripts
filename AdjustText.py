# -*- coding: utf-8 -*-
##AdjustText=name
##Reset=boolean True

from qgis.core import *
from qgis.gui import *
from qgis.utils import *
from PyQt4.QtCore import *
import numpy as np
import math
import sys


def log(msg):
    QgsMessageLog.logMessage(msg, 'MyPlugin', QgsMessageLog.INFO)

def approximately_equal(x1, x2):
    return abs(x2 - x1) < sys.float_info.epsilon * 100

def getFeatureById(layer, featid):
    features = [f for f in layer.getFeatures(QgsFeatureRequest().setFilterFids([featid]))]
    if len(features) != 1:
        return None
    else:
        return features[0]

def move_texts(bboxes):
    for bbox in bboxes:
        featureId = bbox['featureId']
        x = bbox['orgx']
        y = bbox['orgy']
        ha = bbox['ha']
        va = bbox['va']
        set_label_position(featureId, x, y, ha, va)


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


def normalized_point_position(text,extent):
    # [0,1]に正規化されたデータポイントの位置を返す
    xmin, xmax = extent.xMinimum(),extent.xMaximum()
    ymin, ymax = extent.yMinimum(),extent.yMaximum()
    feature = getFeatureById(layer, text.featureId)
    p = feature.geometry().asPoint()
    return ((p[0]-xmin)/(xmax-xmin), (p[1]-ymin)/(ymax-ymin))

def set_bbox(bbox,velocities):
    dx, dy = velocities[0],velocities[1]
    #log("{},{}".format(dx,dy))
    bbox['orgx'] = bbox['orgx']+dx
    bbox['orgy'] = bbox['orgy']+dy
    bbox['xmin'] = bbox['xmin']+dx
    bbox['ymin'] = bbox['ymin']+dy
    bbox['xmax'] = bbox['xmax']+dx
    bbox['ymax'] = bbox['ymax']+dy
    return bbox

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

def transform_coord(bboxes,extent,normalized,random=None):
    xmin, xmax = extent.xMinimum(),extent.xMaximum()
    ymin, ymax = extent.yMinimum(),extent.yMaximum()
    if normalized:
        for i in range(len(bboxes)):
            bboxes[i]['orgx'] = (bboxes[i]['orgx'] -xmin) / (xmax - xmin) + random[i]
            bboxes[i]['orgy'] = (bboxes[i]['orgy'] - ymin) / (ymax - ymin) + random[i]
            bboxes[i]['xmin'] = (bboxes[i]['xmin'] - xmin) / (xmax - xmin) + random[i]
            bboxes[i]['xmax'] = (bboxes[i]['xmax'] - xmin) / (xmax - xmin) + random[i]
            bboxes[i]['ymin'] = (bboxes[i]['ymin'] - ymin) / (ymax - ymin) + random[i]
            bboxes[i]['ymax'] = (bboxes[i]['ymax'] - ymin) / (ymax - ymin) + random[i]
            bboxes[i]['width'] = (bboxes[i]['width'] - xmin) / (xmax - xmin) + random[i]
            bboxes[i]['height'] = (bboxes[i]['height'] - ymin) / (ymax - ymin) + random[i]
            #log("{}".format(bboxes[i]['orgx']))
    else:
        for i in range(len(bboxes)):
            bboxes[i]['orgx'] = bboxes[i]['orgx']*(xmax - xmin)+xmin
            bboxes[i]['orgy'] = bboxes[i]['orgy']*(ymax - ymin)+ymin
            bboxes[i]['xmin'] = bboxes[i]['xmin']*(xmax - xmin)+xmin
            bboxes[i]['xmax'] = bboxes[i]['xmax']*(xmax - xmin)+xmin
            bboxes[i]['ymin'] = bboxes[i]['ymin']*(ymax - ymin)+ymin
            bboxes[i]['ymax'] = bboxes[i]['ymax']*(ymax - ymin)+ymin
            bboxes[i]['width'] = bboxes[i]['width']*(xmax - xmin)+xmin
            bboxes[i]['height'] = bboxes[i]['height']*(ymax - ymin)+ymin
            #log("{}".format(bboxes[i]['orgx']))
    return bboxes

def get_bboxes(texts, expand):
    # ha:left,va:bottomの時のラベル情報を取得
    bboxes = [{'featureId': text.featureId,
               'label': text.labelText ,
               'orgx': text.cornerPoints[0][0] + text.width / 2.0,
               'orgy': text.cornerPoints[0][1] + text.height / 2.0,
               'xmin': text.cornerPoints[0][0] - text.width * (expand[0] - 1) / 2.0,
               'xmax': text.cornerPoints[2][0] + text.width * (expand[0] - 1) / 2.0,
               'ymin': text.cornerPoints[0][1] - text.height * (expand[1] - 1) / 2.0,
               'ymax': text.cornerPoints[2][1] + text.height * (expand[1] - 1) / 2.0,
               'width': text.width * expand[0],
               'height': text.height * expand[1],
               'ha': "center", 'va': "center"}
              for text in texts]
    # for bbox in bboxes:
    #     log("{},xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["label"],bbox["xmin"],bbox["xmax"],bbox["ymin"],bbox["ymax"]))
    return bboxes


def get_dboxes(points, padding):
    # ポイントのpadding付きの範囲を取得
    dboxes = [{'xmin': xy[0] - padding[0],
               'xmax': xy[0] + padding[0],
               'ymin': xy[1] - padding[1],
               'ymax': xy[1] + padding[1],
               }
              for xy in points]
    # for bbox in dboxes:
    #     log("dbox:::xmin:{},xmax:{},ymin:{},ymax:{}".format(bbox["xmin"],bbox["xmax"],bbox["ymin"],bbox["ymax"]))
    return dboxes



def get_midpoint(bbox):
    cx = (bbox["xmin"] + bbox["xmax"]) / 2
    cy = (bbox["ymin"] + bbox["ymax"]) / 2
    return np.array([cx, cy])

def overlaps(bbox1, bbox2):
    x0 = np.maximum(bbox1['xmin'], bbox2['xmin'])
    x1 = np.minimum(bbox1['xmax'], bbox2['xmax'])
    y0 = np.maximum(bbox1['ymin'], bbox2['ymin'])
    y1 = np.minimum(bbox1['ymax'], bbox2['ymax'])
    return True if x0 <= x1 and y0 <= y1 else False

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

def put_within_bounds(bbox, xlim, ylim):
    if bbox["xmin"] < xlim[0]:
        bbox = set_bbox(bbox,[xlim[0] - bbox["xmin"],0])
    elif bbox["xmax"] > xlim[1]:
        bbox = set_bbox(bbox, [xlim[1] - bbox["xmax"], 0])
    if bbox["ymin"] < ylim[0]:
        bbox = set_bbox(bbox, [0,ylim[0] - bbox["ymin"]])
    elif bbox["ymax"] > ylim[1]:
        bbox = set_bbox(bbox, [0,ylim[1] - bbox["ymax"]])

    return bbox


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
    lr = canvas.labelingResults()
    extent = canvas.extent()
    texts = lr.labelsWithinRect(extent)

    n_texts = len(texts)
    r = np.random.normal(0, force_push, n_texts)
    bboxes = get_bboxes(texts, expand=(1.05, 1.2))
    bboxes = transform_coord(bboxes,extent,normalized=True,random=r)
    xbounds = [0, 1]
    ybounds = [0, 1]

    #点の座標
    points = np.array([normalized_point_position(text,extent) for text in texts])
    dboxes = get_dboxes(points, padding=(1e-2, 1e-2))
    n_points = len(points)
    orig_cent = np.array([(bbox['orgx'],bbox['orgy']) for bbox in bboxes])
    #log("len:{}".format(len(points)))
    velocities = np.array([[0.0,0.0]]*n_texts)
    velocity_decay = 0.7
    iter = 0
    any_overlaps = True
    i_overlaps = True

    while (any_overlaps and iter < maxiter):
        #log("iter:{}".format(iter))
        iter = iter + 1
        any_overlaps = False
        # The forces get weaker over time.
        force_push = force_push * 0.99999
        force_pull = force_pull * 0.9999
        for i in range(n_texts):
            i_overlaps = False
            f = np.array([0, 0])
            #ラベルの中心座標
            ci = get_midpoint(bboxes[i])
            #log("target:{}".format(bboxes[i]['label']))
            for j in range(n_points):
                if i == j:
                    # 自分自身の点とラベルが重なるなら.
                    if overlaps(dboxes[i], bboxes[i]):
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, points[i], force_push)
                        #log("A:{}".format(f))
                else:
                    cj = get_midpoint(bboxes[j])
                    # ラベル同士が重なる場合.
                    if j < n_texts and overlaps(bboxes[i], bboxes[j]):
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, cj, force_push)
                        #log("B:{}".format(f))
                    # ラベルが他の点と重なる場合.
                    if overlaps(dboxes[j], bboxes[i]):
                        any_overlaps = True
                        i_overlaps = True
                        f = f + repel_force(ci, points[j], force_push)
                        #log("C:{}".format(f))
            # ラベルがどこにも重なっていないなら、最初のラベルの位置にちょっと戻る.
            if not i_overlaps:
                f = f + spring_force(orig_cent[i], ci, force_pull)
                #log("{},{}".format(orig_xy[i], ci))
                #log("D:{}".format(f))

            velocities[i] = velocities[i] * velocity_decay + f
            #log("vel:{},f:{}".format(velocities[i],f))
            #log("before:{}".format(bboxes[i]))
            bboxes[i] = set_bbox(bboxes[i], velocities[i])
            #log("after:{}".format(bboxes[i]))
            # Put boxes within bounds
            bboxes[i] = put_within_bounds(bboxes[i], xbounds, ybounds)
            # look for line clashes

            if not any_overlaps or iter % 5 == 0:
                for j in range(n_points):
                    cj = get_midpoint(bboxes[j])
                    ci = get_midpoint(bboxes[i])
                    # Switch label positions if lines overlap
                    if i != j and j < n_texts and line_intersect(ci, points[i], cj, points[j]):
                        log("interA")
                        any_overlaps = True
                        bboxes[i] = set_bbox(bboxes[i],spring_force(cj, ci, 1))
                        bboxes[j] = set_bbox(bboxes[j],spring_force(ci, cj, 1))
                        # Check if resolved
                        ci = get_midpoint(bboxes[i])
                        cj = get_midpoint(bboxes[j])
                        if line_intersect(ci, points[i], cj, points[j]):
                            log("interB")
                            bboxes[i] = set_bbox(bboxes[i],spring_force(cj, ci, 1.25))
                            bboxes[j] = set_bbox(bboxes[j],spring_force(ci, cj, 1.25))


    bboxes = transform_coord(bboxes,extent,normalized=False)
    move_texts(bboxes)


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


def get_features_within_canvas():
    region = canvas.extent().asWktPolygon()
    where_text = u"contains(geom_from_wkt('{0}'), $geometry)".format(region)
    expression = QgsExpression(where_text)
    features = layer.getFeatures(QgsFeatureRequest(expression))
    return features


def reset_labelLayer():
    if layer is not None:
        features = get_features_within_canvas()
        if len(list(features)) == 0:
            return
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