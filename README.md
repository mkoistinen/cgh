# Readme


## Coordinate System

Imagine the film-plate ("plate")--an infinitely thin, rectangular shape--is
situated on top of a optical bench oriented upright, standing on its edge.

We will define the coordinate system relative to this plate. The viewing side
of the hologram will be the side opposite of the plate that is opposite the
object. When facing this plate as the viewer, left-to-right is the X-axis,
top-to-bottom is the Y-axis and the Z-axis passes through the center of the
plate, perpendicularly.

All coordinates are relative to the point 0, 0, 0 located at the center of the
plate.


## Optical Configuration

Since we are computing Transmission Holograms, the object, its reflected light
field and the reference field are positioned on the opposite side of the plate.

We will model the position of the object with coordinates relative to the
center of the plate. Without any translation, the object would be sitting with
its local center positioned directly in the middle of the plate (not useful!)

An increase in the Z coordinate will place the object further "behind" the
plate (more useful).


### Reference Field


The reference field is the light field that illuminates the plate directly.
This field never occludes nor illuminates the object. It interferes with the
object's reflected field only at the film plate itself. The reference field is
always a spherical wave field, but a planar wave field can be simulated when
its origin's distance from the center of the plate is 0.0. In this case, the
distance is set to a distance similar to the distance of the Sun from Earth and
is therefore effectively planar.


### Illumination Field

The object is illuminated by an illumination field. This is also a spherical
light source that can be made effectively planar by configuring its distance to
the center of the plate to 0.0. This field never directly illuminates the
plate, only the object.


### Object Field

The object field is the light field that emanates as reflection of the
illumination field off the object towards the film plate.
