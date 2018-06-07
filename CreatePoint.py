##CreatePoint=name
##Lat=number 35
##Lon=number 135

from qgis.core import QgsVectorLayer, QgsFeature, QgsPoint, QgsGeometry, QgsMapLayerRegistry

mem_layer = QgsVectorLayer("Point", "temporary_points", "memory")
prov = mem_layer.dataProvider()
new_feat = QgsFeature()
new_feat.setGeometry(QgsGeometry.fromPoint(QgsPoint(Lon,Lat)))
prov.addFeatures([new_feat])
QgsMapLayerRegistry.instance().addMapLayer(mem_layer)
