"""Blender script to import SAM 3D Body OBJ sequence + camera.
Paste into Blender's Scripting tab and run.

Camera is static at origin (focal baked from video resolution).
Mesh animates via shape keys — one per frame.
"""
import bpy
import json
from pathlib import Path

# ============ CONFIGURE THESE ============
OBJ_DIR = r".\\_test_obj"
CAMERA_JSON = r".\\_test_obj\\camera.json"
# =========================================

with open(CAMERA_JSON) as f:
    cam_data = json.load(f)

fps = cam_data["fps"]
img_w = cam_data["image_width"]
img_h = cam_data["image_height"]
lens_mm = cam_data["lens_mm"]
sensor_w = cam_data["sensor_width_mm"]

# Scene setup
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.render.resolution_x = img_w
bpy.context.scene.render.resolution_y = img_h

# Static camera at origin, looking -Z (Blender default for new camera)
cam = bpy.data.cameras.new("SAM3D_Camera")
cam.lens = lens_mm
cam.sensor_width = sensor_w
cam.sensor_fit = 'HORIZONTAL'
cam_obj = bpy.data.objects.new("SAM3D_Camera", cam)
bpy.context.collection.objects.link(cam_obj)
cam_obj.location = (0, 0, 0)
cam_obj.rotation_euler = (0, 0, 0)
bpy.context.scene.camera = cam_obj

# Import OBJ sequence as shape keys
obj_files = sorted(Path(OBJ_DIR).glob("frame_*_p0.obj"))
if not obj_files:
    print("No OBJ files found!")
    raise SystemExit

bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(obj_files) - 1

# First frame = base mesh
bpy.ops.wm.obj_import(filepath=str(obj_files[0]))
base_obj = bpy.context.selected_objects[0]
base_obj.name = "SAM3D_Body"
base_obj.shape_key_add(name="Basis")

for i, obj_file in enumerate(obj_files[1:], 1):
    bpy.ops.wm.obj_import(filepath=str(obj_file))
    imported = bpy.context.selected_objects[0]

    sk = base_obj.shape_key_add(name=f"frame_{i:04d}")
    for vi, vert in enumerate(imported.data.vertices):
        sk.data[vi].co = vert.co
    bpy.data.objects.remove(imported, do_unlink=True)

    # Animate: key on at this frame, off at neighbors
    sk.value = 0.0
    sk.keyframe_insert(data_path="value", frame=i - 1)
    sk.value = 1.0
    sk.keyframe_insert(data_path="value", frame=i)
    sk.value = 0.0
    sk.keyframe_insert(data_path="value", frame=i + 1)

print(f"Imported {len(obj_files)} frames. Camera lens={lens_mm:.1f}mm")
