##MergeLines=name

from qgis.core import QgsVectorLayer, QgsFeature, QgsPoint, QgsRectangle, QgsGeometry, QgsMapLayerRegistry,QgsMessageLog
from qgis.utils import iface
from qgis.gui import QgsMessageBar
from PyQt4.QtCore import QSettings
import math

def distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

layer = iface.activeLayer()
if layer.wkbType()==2:
    selected_features = layer.selectedFeatures()
    if len(selected_features)==2:
        f0 = selected_features[0]
        f1 = selected_features[1]
        line0 = f0.geometry().asPolyline()
        line1 = f1.geometry().asPolyline()
        dist = [distance(li0,li1) for li0,li1 in [(line0[-1],line1[0]),(line0[0], line1[-1]),(line0[0], line1[0]),(line0[-1], line1[-1])]]
        type = dist.index(min(dist))
        if type==0:
            pass
        elif type==1:
            line0.reverse()
            line1.reverse()
        elif type==2:
            line0.reverse()
        elif type==3:
            line1.reverse()
        line = line0 + line1[1:]
        geom = QgsGeometry.fromPolyline(line)

        layer.beginEditCommand("Feature merged")
        settings = QSettings()
        disable_attributes = settings.value("/qgis/digitizing/disable_enter_attribute_values_dialog", False, type=bool)
        if disable_attributes:
            layer.changeGeometry(f0.id(), geom)
            layer.deleteFeature(f1.id())
            layer.endEditCommand()
        else:
            dlg = iface.getFeatureForm(layer, f0)
            if dlg.exec_():
                layer.changeGeometry(f0.id(), geom)
                layer.deleteFeature(f1.id())
                layer.endEditCommand()
            else:
                layer.destroyEditCommand()
        iface.mapCanvas().refresh()
    else:
        iface.messageBar().pushMessage("Warning", "Select two feature!",
                                       level=QgsMessageBar.WARNING)
else:
    iface.messageBar().pushMessage("Warning", "Only Support LineString.",
                                   level=QgsMessageBar.WARNING)


