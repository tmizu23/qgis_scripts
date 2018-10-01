##AdjustText=name

from qgis.core import *
from qgis.gui import *
from qgis.utils import *
from PyQt4.QtCore import *
from PyQt4.QtGui import *

def log(msg):
    QgsMessageLog.logMessage(msg, 'MyPlugin', QgsMessageLog.INFO)

def getFeatureById(layer,featid):
    features = [f for f in layer.getFeatures(QgsFeatureRequest().setFilterFids([featid]))]
    if len(features) != 1:
        return None
    else:
        return features[0]

canvas=iface.mapCanvas()

layer = iface.activeLayer()
features = layer.selectedFeatures()
if len(features)==0:
    features = layer.getFeatures()
# layer.startEditing()
# for feature in features:
#     p = feature.geometry().asPoint()
#     x = feature['x']
#     feature['x'] = x+100
#     layer.updateFeature(feature)
#     log("{}".format(x))
# layer.commitChanges()

lr = canvas.labelingResults()
extent = canvas.extent()
for lrl in lr.labelsWithinRect(extent):
    if lrl.layerID == layer.id():
        angle = 90 - QgsPoint(lrl.cornerPoints[0]).azimuth(QgsPoint(lrl.cornerPoints[1]))
        xmin = lrl.cornerPoints[0][0]
        xmax = lrl.cornerPoints[2][0]
        ymin = lrl.cornerPoints[0][1]
        ymax = lrl.cornerPoints[2][1]
        rlabel = angle
        featid = lrl.featureId
        feature=getFeatureById(layer,featid)
        p = feature.geometry().asPoint()
        log("{},{},{},{},{},{},{},{}".format(featid,p[0],p[1],xmin,xmax,ymin,ymax,angle))
