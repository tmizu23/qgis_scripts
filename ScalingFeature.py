##ScalingFeature=name
##Scale=number 2.0

from qgis.core import QgsVectorLayer, QgsFeature, QgsPoint, QgsRectangle, QgsGeometry, QgsMapLayerRegistry,QgsMessageLog
from qgis.utils import iface
from qgis.gui import QgsMessageBar


def scalingPoint(cent,seg,scale):
    w = scale * (seg[0] - cent[0])
    h = scale * (seg[1] - cent[1])
    p = QgsPoint(cent[0] + w, cent[1] + h)
    return p

canvas=iface.mapCanvas()

layer = iface.activeLayer()
features = layer.selectedFeatures()
if len(features)==0:
    features = layer.getFeatures()

epsg = layer.crs().postgisSrid()
if layer.wkbType()==2:
    geomtype="LineString"
elif layer.wkbType()==3:
    geomtype="Polygon"
#QgsMessageLog.logMessage("{}".format(geomtype), "name")
if geomtype=="LineString" or geomtype=="Polygon":
    uri = geomtype + "?crs=epsg:" + str(epsg)
    mem_layer = QgsVectorLayer(uri, 'scaling_layer', 'memory')
    prov = mem_layer.dataProvider()

    fields=layer.dataProvider().fields().toList()
    mem_layer.dataProvider().addAttributes(fields)
    mem_layer.updateFields()

    scaling_feature=[]
    for feature in features:
        new_feat = QgsFeature()
        new_feat.setAttributes(feature.attributes())
        if geomtype == "LineString":
            geom = feature.geometry().asPolyline()
        elif geomtype == "Polygon":
            geom = feature.geometry().asPolygon()[0]
        cent = feature.geometry().centroid().asPoint()
        points=[]
        for seg in geom:
            p = scalingPoint(cent,seg,Scale)
            points.append(p)
        if geomtype=="LineString":
            new_feat.setGeometry(QgsGeometry.fromPolyline(points))
        elif geomtype=="Polygon":
            new_feat.setGeometry(QgsGeometry.fromPolygon([points]))
        scaling_feature.append(new_feat)

    prov.addFeatures(scaling_feature)
    QgsMapLayerRegistry.instance().addMapLayer(mem_layer)
else:
    iface.messageBar().pushMessage("Warning", "No support layer type. Support LineString or Polygon.",
                                   level=QgsMessageBar.WARNING)
