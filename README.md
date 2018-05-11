# qgis_scripts
This is scripts for QGIS

## RectangleArea
create rectangle polygon by scale and paper size(cm).  
if scale 25000, Height 20cm, Width 20cm, then 5km x 5km rectangle is created at center of map canvas.   

## ScalingFeature
create scaled feature from each feature centrid of selected layer.  
if scale 2.0, memory layer is created having feature size is double.
only support LineString and Polygon.  
