##RetangleArea=name
##Scale=number 25000
##Height=number 20
##Width=number 20

from qgis.core import QgsVectorLayer, QgsFeature, QgsPoint, QgsRectangle, QgsGeometry, QgsMapLayerRegistry
from qgis.utils import iface
from qgis.gui import QgsMessageBar

canvas=iface.mapCanvas()
if canvas.mapRenderer().destinationCrs().projectionAcronym() != "longlat":
    epsg = canvas.mapRenderer().destinationCrs().postgisSrid()
    #QgsMessageLog.logMessage("{}".format(epsg), "name")
    uri = "Polygon?crs=epsg:" + str(epsg) + "&field=id:integer"
    mem_layer = QgsVectorLayer(uri, 'rectangular_area', 'memory')
    prov = mem_layer.dataProvider()

    x=Width*Scale/100
    y=Height*Scale/100

    p = canvas.extent().center()
    new_feat = QgsFeature()
    new_feat.setAttributes([0])
    p1 = QgsPoint(p[0]-x/2.0, p[1]-y/2.0)
    p2 = QgsPoint(p[0]+x/2.0, p[1]+y/2.0)
    new_ext = QgsRectangle(p1,p2)
    new_tmp_feat = new_ext.asWktPolygon()
    new_feat.setGeometry(QgsGeometry.fromWkt(new_tmp_feat))
    prov.addFeatures([new_feat])

    QgsMapLayerRegistry.instance().addMapLayer(mem_layer)
else:
    iface.messageBar().pushMessage("Warning", "Change CRS from latlon", level=QgsMessageBar.WARNING)