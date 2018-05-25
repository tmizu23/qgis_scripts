# qgis_scripts
This is scripts for QGIS

## RectangleArea
create rectangle polygon by scale and paper size(cm).  
if scale 25000, Height 20cm, Width 20cm, then 5km x 5km rectangle is created at center of map canvas.   

## ScalingFeature
create scaled feature from each feature centrid of selected layer.  
if scale 2.0, temp layer is created having each feature size is double.
only support LineString and Polygon.  

## MergeLines
merge splited two lines to one.
select two line, then execute script.
the first point of the second line will be deleted before merged.   
only support LineString.

