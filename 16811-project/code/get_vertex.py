# print out selected vertices from blender

import bpy
import bmesh
obj = bpy.context.edit_object
me = obj.data
bm = bmesh.from_edit_mesh(me)
bm.faces.active = None

inds = []
for v in bm.verts:
    if v.select:
        inds.append(v.index)