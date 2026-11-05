"""Renders segment meshes for pipeline.movie_segments, inside Blender (not imported by the pipeline):

    blender -b --factory-startup --python pipeline/blender_segments.py -- job.json

job.json: size [w, h], samples, origin (s0 voxels, subtracted from everything), meshes [{path (PLY, s0
voxels), color [r, g, b]}], frames [{path (PNG out), camera {matrix (4 x 4 rows), lens, sensor}, clip [lo, hi]
(world z, shifted: only the meshes' part between these is shown)}]. Cycles (on an NVIDIA GPU if there is
one: OptiX, else CUDA; else the CPU), lit by a key and a fill light
that move with the camera and a grey sky; a transparent background (RGBA, straight alpha); where the clip
opens a mesh, its inside shows as a flat, darker cap.
"""

import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
for kind in ("OPTIX", "CUDA"):                       # an NVIDIA GPU if there is one
    try:
        prefs.compute_device_type = kind
    except TypeError:                                # (this build lacks the backend)
        continue
    prefs.get_devices()
    if any(d.type == kind for d in prefs.devices):
        for d in prefs.devices:
            d.use = d.type == kind
        scene.cycles.device = "GPU"
        break
print(f"Cycles device: {scene.cycles.device} ({prefs.compute_device_type})", flush=True)
scene.cycles.samples = int(job["samples"])
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 4
scene.cycles.transparent_max_bounces = 64
scene.render.use_persistent_data = True               # the scene (BVH) kept between frames
scene.render.film_transparent = True
scene.render.resolution_x, scene.render.resolution_y = job["size"]
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"       # the colours as given

world = bpy.data.worlds.new("sky")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.38, 1.0)
scene.world = world

cam_obj = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.data.sensor_fit = "AUTO"                       # the long side
cam_obj.data.clip_start, cam_obj.data.clip_end = 1.0, 1e7


def light(name, energy, direction):
    """A sun moving with the camera, shining along ``direction`` in camera space (x right, y up, z back)."""
    obj = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
    obj.data.energy, obj.data.angle = energy, math.radians(15)
    obj.parent = cam_obj
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(Vector(direction).normalized())
    scene.collection.objects.link(obj)


light("key", 3.0, (0.5, -0.6, -1.0))                  # from the upper left, in front
light("fill", 0.8, (-0.7, 0.2, -0.6))                 # weaker, from the right

clips = []
origin = job["origin"]
for i, mesh in enumerate(job["meshes"]):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.ply_import(filepath=mesh["path"], forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.location = (-origin[0], -origin[1], -origin[2])
    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
    mat = bpy.data.materials.new(f"segment{i}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    color = (*mesh["color"], 1.0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.45
    cap = nodes.new("ShaderNodeEmission")               # the inside, where the clip opens the mesh
    cap.inputs["Color"].default_value = tuple(0.55 * c for c in mesh["color"]) + (1.0,)
    geo = nodes.new("ShaderNodeNewGeometry")
    surface = nodes.new("ShaderNodeMixShader")
    links.new(geo.outputs["Backfacing"], surface.inputs[0])
    links.new(bsdf.outputs[0], surface.inputs[1])
    links.new(cap.outputs[0], surface.inputs[2])
    xyz = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Position"], xyz.inputs[0])
    lo, hi = nodes.new("ShaderNodeValue"), nodes.new("ShaderNodeValue")
    above, below, both = (nodes.new("ShaderNodeMath") for _ in range(3))
    above.operation, below.operation, both.operation = "GREATER_THAN", "LESS_THAN", "MULTIPLY"
    links.new(xyz.outputs["Z"], above.inputs[0])
    links.new(lo.outputs[0], above.inputs[1])
    links.new(xyz.outputs["Z"], below.inputs[0])
    links.new(hi.outputs[0], below.inputs[1])
    links.new(above.outputs[0], both.inputs[0])
    links.new(below.outputs[0], both.inputs[1])
    clear = nodes.new("ShaderNodeBsdfTransparent")
    shown = nodes.new("ShaderNodeMixShader")
    links.new(both.outputs[0], shown.inputs[0])
    links.new(clear.outputs[0], shown.inputs[1])
    links.new(surface.outputs[0], shown.inputs[2])
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(shown.outputs[0], out.inputs["Surface"])
    obj.data.materials.append(mat)
    clips.append((lo, hi))

for k, frame in enumerate(job["frames"]):
    cam = frame["camera"]
    cam_obj.matrix_world = Matrix(cam["matrix"])
    cam_obj.data.lens, cam_obj.data.sensor_width = cam["lens"], cam["sensor"]
    for lo, hi in clips:
        lo.outputs[0].default_value, hi.outputs[0].default_value = frame["clip"]
    scene.render.filepath = frame["path"]
    bpy.ops.render.render(write_still=True)
    print(f"mesh layer {k + 1}/{len(job['frames'])}: {frame['path']}", flush=True)
