"""
SAM 3D Body — Video Processing Script
Two-pass pipeline:
  Pass 1: Run inference on all frames, collect raw predictions
  Pass 2: Apply Savitzky-Golay temporal smoothing, then render + export

Outputs:
  1. Visualization video (original + skeleton + mesh overlay + side view)
  2. Per-frame .obj files with UV layout (optional, --export_obj)
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

# obj2abc binary from headcase — check common locations
_OBJ2ABC_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "..", "ML_repos", "headcase", "alembic_cpp", "bin", "obj2abc.exe"),
    os.path.join(os.path.dirname(__file__), "..", "..", "ML_repos", "headcase", "bin", "obj2abc.exe"),
    os.path.join(os.path.dirname(__file__), "..", "..", "ML_repos", "headcase", "alembic_cpp", "build", "Release", "obj2abc.exe"),
]
OBJ2ABC_PATH = next((os.path.normpath(p) for p in _OBJ2ABC_CANDIDATES if os.path.exists(p)), "obj2abc.exe")

import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer
from tools.vis_utils import visualize_sample_together


# ---------------------------------------------------------------------------
# Savitzky-Golay temporal smoothing (adapted from headcase filter_smooth.py)
# ---------------------------------------------------------------------------

def _fit_window(window_length, poly_order, n_frames):
    """Clamp savgol window to fit available frames."""
    if n_frames < window_length:
        window_length = n_frames
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= poly_order:
        return 0  # signal: can't smooth
    return window_length


def smooth_array_savgol(data, window_length=11, poly_order=3):
    """Apply savgol_filter per-dimension across time axis.

    Args:
        data: (N, D) array where N=frames, D=dimensions
        window_length: must be odd, >= poly_order+1
        poly_order: polynomial order for the filter
    Returns:
        Smoothed (N, D) array
    """
    n_frames = data.shape[0]
    window_length = _fit_window(window_length, poly_order, n_frames)
    if window_length == 0:
        return data

    smoothed = np.zeros_like(data)
    if data.ndim == 1:
        smoothed = savgol_filter(data, window_length, poly_order)
    else:
        for dim in range(data.shape[1]):
            smoothed[:, dim] = savgol_filter(data[:, dim], window_length, poly_order)
    return smoothed


def smooth_vertices_savgol(all_verts, window_length=11, poly_order=3, chunk_size=2000):
    """Smooth vertex positions across time. Processes in chunks for memory.

    Args:
        all_verts: (N, V, 3) array — N frames, V vertices, 3 coords
        window_length, poly_order: savgol parameters
        chunk_size: vertices per chunk to limit memory
    Returns:
        Smoothed (N, V, 3) array
    """
    n_frames, n_verts, n_coords = all_verts.shape
    window_length = _fit_window(window_length, poly_order, n_frames)
    if window_length == 0:
        return all_verts

    smoothed = np.zeros_like(all_verts)

    for start in range(0, n_verts, chunk_size):
        end = min(start + chunk_size, n_verts)
        chunk = all_verts[:, start:end, :]  # (N, chunk, 3)

        for v in range(chunk.shape[1]):
            for c in range(n_coords):
                smoothed[:, start + v, c] = savgol_filter(
                    chunk[:, v, c], window_length, poly_order
                )

    return smoothed


def apply_temporal_smoothing(all_outputs, window_length=11, poly_order=3):
    """Apply Savitzky-Golay smoothing to per-frame predictions.

    Smooths: pred_vertices, pred_cam_t, pred_keypoints_2d, pred_keypoints_3d
    Only handles single-person tracking (person 0 per frame).
    Frames with no detections are left as-is (empty list).
    """
    # Find valid frames (frames with at least one detection)
    valid_indices = [i for i, out in enumerate(all_outputs) if len(out) > 0]

    if len(valid_indices) < 3:
        print(f"  Only {len(valid_indices)} valid frames — skipping smoothing")
        return all_outputs

    print(f"  Smoothing {len(valid_indices)} valid frames (window={window_length}, poly={poly_order})")

    # --- Collect arrays from valid frames (person 0 only) ---
    verts_list = []
    cam_t_list = []
    kp2d_list = []
    kp3d_list = []

    for idx in valid_indices:
        person = all_outputs[idx][0]
        verts_list.append(person["pred_vertices"])
        cam_t_list.append(person["pred_cam_t"])
        kp2d_list.append(person["pred_keypoints_2d"])
        kp3d_list.append(person["pred_keypoints_3d"])

    verts_arr = np.array(verts_list)     # (N, 18439, 3)
    cam_t_arr = np.array(cam_t_list)     # (N, 3)
    kp2d_arr = np.array(kp2d_list)       # (N, 70, 2)
    kp3d_arr = np.array(kp3d_list)       # (N, 70, 3)

    # --- Smooth ---
    print("  Smoothing vertices...")
    verts_smooth = smooth_vertices_savgol(verts_arr, window_length, poly_order)

    print("  Smoothing camera translation...")
    cam_t_smooth = smooth_array_savgol(cam_t_arr, window_length, poly_order)

    print("  Smoothing 2D keypoints...")
    kp2d_flat = kp2d_arr.reshape(len(valid_indices), -1)
    kp2d_smooth = smooth_array_savgol(kp2d_flat, window_length, poly_order)
    kp2d_smooth = kp2d_smooth.reshape(kp2d_arr.shape)

    print("  Smoothing 3D keypoints...")
    kp3d_flat = kp3d_arr.reshape(len(valid_indices), -1)
    kp3d_smooth = smooth_array_savgol(kp3d_flat, window_length, poly_order)
    kp3d_smooth = kp3d_smooth.reshape(kp3d_arr.shape)

    # --- Write back ---
    for i, idx in enumerate(valid_indices):
        all_outputs[idx][0]["pred_vertices"] = verts_smooth[i]
        all_outputs[idx][0]["pred_cam_t"] = cam_t_smooth[i]
        all_outputs[idx][0]["pred_keypoints_2d"] = kp2d_smooth[i]
        all_outputs[idx][0]["pred_keypoints_3d"] = kp3d_smooth[i]

    print("  Smoothing complete.")
    return all_outputs


# ---------------------------------------------------------------------------
# OBJ + camera export
# ---------------------------------------------------------------------------

def transform_verts_to_world(vertices, cam_t):
    """Transform model-space vertices to world space matching the renderer.

    The pyrender renderer does:
      1. Rotate mesh 180° around X  (Y *= -1, Z *= -1)
      2. Place camera at [-cam_t.x, cam_t.y, cam_t.z]

    To bake this into the OBJ so it lines up with a camera at the origin
    looking down -Z (standard Blender/Houdini convention):
      - Apply the 180° X rotation to vertices
      - Offset by camera translation (so mesh is positioned relative to cam)
    """
    world = vertices.copy()
    # 180° rotation around X: Y and Z flip sign
    world[:, 1] *= -1.0
    world[:, 2] *= -1.0
    # Subtract camera translation to move camera to origin.
    # Renderer places camera at [-cam_t[0], cam_t[1], cam_t[2]].
    # mesh_in_camera_space = mesh_world - camera_pos
    world[:, 0] -= -cam_t[0]   # i.e. += cam_t[0]
    world[:, 1] -= cam_t[1]
    world[:, 2] -= cam_t[2]
    return world


def export_obj(filepath, vertices, faces, texcoords=None, texcoord_faces=None):
    """Export mesh as .obj with UV layout."""
    with open(filepath, 'w') as f:
        f.write(f"# SAM 3D Body mesh export\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")

        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        if texcoords is not None:
            f.write(f"\n")
            for uv in texcoords:
                f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")

        f.write(f"\n")
        if texcoords is not None and texcoord_faces is not None:
            for fi, ti in zip(faces, texcoord_faces):
                f.write(f"f {fi[0]+1}/{ti[0]+1} {fi[1]+1}/{ti[1]+1} {fi[2]+1}/{ti[2]+1}\n")
        else:
            for fi in faces:
                f.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


def export_camera_json(filepath, focal_length, fps, img_width, img_height):
    """Export static camera data as JSON for Blender/Houdini import.

    Camera convention: located at origin, looking down -Z, Y-up.
    Meshes are pre-transformed to world space relative to this camera.

    The focal_length is sqrt(w²+h²) — constant for a given resolution.
    To get Blender lens_mm: focal_px * 36 / image_width
    """
    sensor_w = 36.0
    lens_mm = focal_length * sensor_w / img_width

    out = {
        "format": "sam3d_camera",
        "version": 2,
        "fps": fps,
        "image_width": img_width,
        "image_height": img_height,
        "camera_model": "pinhole",
        "focal_length_px": focal_length,
        "sensor_width_mm": sensor_w,
        "lens_mm": lens_mm,
        "principal_point": [img_width / 2.0, img_height / 2.0],
        "note": "Static camera at origin looking -Z, Y-up. OBJs have cam translation baked into vertices.",
    }
    with open(filepath, 'w') as f:
        json.dump(out, f, indent=2)


def export_sidecar(filepath, all_outputs, fps, width, height, focal_length):
    """Export raw MHR predictions as sidecar JSON for AE plugin import.

    Analogous to headcase's *_sidecar.json — the AE plugin reads this,
    runs MHR forward on the fly, and lets users delete/interpolate keyframes.
    """
    persons = {}  # person_id -> per-frame data

    for frame_idx, outputs in enumerate(all_outputs):
        for pid, person in enumerate(outputs):
            pid_str = str(pid)
            if pid_str not in persons:
                persons[pid_str] = {
                    "valid_frames": [],
                    "pred_cam_t": [],
                    "bbox": [],
                    "pred_keypoints_2d": [],
                    "mhr_params": {
                        "mhr_model_params": [],
                        "shape_params": [],
                        "global_rot": [],
                        "body_pose_params": [],
                        "hand_pose_params": [],
                        "scale_params": [],
                        "expr_params": [],
                    },
                }
            p = persons[pid_str]
            p["valid_frames"].append(frame_idx)
            p["pred_cam_t"].append(person["pred_cam_t"].tolist())
            p["bbox"].append(person["bbox"].tolist())
            p["pred_keypoints_2d"].append(person["pred_keypoints_2d"].tolist())

            mp = p["mhr_params"]
            mp["mhr_model_params"].append(person["mhr_model_params"].tolist())
            mp["shape_params"].append(person["shape_params"].tolist())
            mp["global_rot"].append(person["global_rot"].tolist())
            mp["body_pose_params"].append(person["body_pose_params"].tolist())
            mp["hand_pose_params"].append(person["hand_pose_params"].tolist())
            mp["scale_params"].append(person["scale_params"].tolist())
            mp["expr_params"].append(person["expr_params"].tolist())

    sensor_w = 36.0
    lens_mm = focal_length * sensor_w / width

    sidecar = {
        "format": "sam3d_body_sidecar",
        "format_version": "1.0",
        "video_info": {
            "resolution": {"width": width, "height": height},
            "fps": fps,
            "total_frames": len(all_outputs),
        },
        "camera": {
            "model": "pinhole",
            "focal_length_px": focal_length,
            "sensor_width_mm": sensor_w,
            "lens_mm": lens_mm,
            "principal_point": [width / 2.0, height / 2.0],
        },
        "persons_data": persons,
    }

    with open(filepath, 'w') as f:
        json.dump(sidecar, f)
    # Report size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    n_persons = len(persons)
    n_frames = max((len(p["valid_frames"]) for p in persons.values()), default=0)
    print(f"  Sidecar: {filepath} ({size_mb:.1f} MB, {n_persons} person(s), {n_frames} frames)")


def export_alembic(abc_path, template_obj_path, all_world_verts, fps, n_verts, camera_json_path=None):
    """Export animated mesh as Alembic via obj2abc --stream.

    Pipes binary vertex data (HCAS format) to the headcase obj2abc tool.
    Template OBJ provides topology + UVs (written once).
    Vertex positions animate per-frame.

    Args:
        abc_path: output .abc file path
        template_obj_path: first-frame OBJ with topology/UVs
        all_world_verts: list of (N_verts, 3) arrays, one per valid frame
        fps: frame rate
        n_verts: vertex count per frame
        camera_json_path: optional path to camera JSON (baked into .abc)
    """
    if not os.path.exists(OBJ2ABC_PATH):
        print(f"  WARNING: obj2abc not found at {OBJ2ABC_PATH}")
        print(f"  Skipping Alembic export. Build obj2abc from headcase/alembic_cpp/")
        return False

    n_frames = len(all_world_verts)
    print(f"  Piping {n_frames} frames ({n_verts} verts) to obj2abc...")

    # Build HCAS binary header (32 bytes)
    header = struct.pack(
        '<4sIIIfII I',  # little-endian: magic, version, frameCount, vertexCount, fps, startFrame, flags, reserved
        b'HCAS',        # magic
        1,              # version
        n_frames,       # frameCount
        n_verts,        # vertexCount
        fps,            # fps
        1,              # startFrame (1-based)
        0,              # flags
        0,              # reserved
    )

    cmd = [
        OBJ2ABC_PATH,
        "--stream",
        "--template", template_obj_path,
        "--no-normals",
    ]
    if camera_json_path and os.path.exists(camera_json_path):
        cmd += ["--camera-json", camera_json_path]
    cmd.append(abc_path)

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    # Write header
    proc.stdin.write(header)

    # Write per-frame vertex data as flat float32 arrays
    for verts in all_world_verts:
        proc.stdin.write(verts.astype(np.float32).tobytes())

    proc.stdin.close()
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        print(f"  obj2abc FAILED (code {proc.returncode})")
        if stderr:
            print(f"  stderr: {stderr.decode('utf-8', errors='replace')}")
        return False

    if stdout:
        # Print obj2abc output (progress info)
        for line in stdout.decode('utf-8', errors='replace').strip().split('\n'):
            print(f"  {line}")

    return True


BLENDER_IMPORT_SCRIPT = '''\
"""Blender script to import SAM 3D Body OBJ sequence + camera.
Paste into Blender's Scripting tab and run.

Camera is static at origin (focal baked from video resolution).
Mesh animates via shape keys — one per frame.
"""
import bpy
import json
from pathlib import Path

# ============ CONFIGURE THESE ============
OBJ_DIR = r"{obj_dir}"
CAMERA_JSON = r"{camera_json}"
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

    sk = base_obj.shape_key_add(name=f"frame_{{i:04d}}")
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

print(f"Imported {{len(obj_files)}} frames. Camera lens={{lens_mm:.1f}}mm")
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM 3D Body Video Processor")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", default="", help="Output video path (default: input_sam3d.mp4)")
    parser.add_argument("--checkpoint", default="../models/model.ckpt", help="Model checkpoint path")
    parser.add_argument("--mhr_path", default="../models/mhr_model.pt", help="MHR model path")
    parser.add_argument("--export_obj", action="store_true", help="Export per-frame .obj files")
    parser.add_argument("--export_abc", action="store_true", help="Export Alembic .abc (animated mesh + UVs)")
    parser.add_argument("--export_sidecar", action="store_true", help="Export raw MHR params as sidecar JSON (for AE plugin)")
    parser.add_argument("--obj_dir", default="", help="Output directory for .obj/.abc files")
    parser.add_argument("--bbox_thresh", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--no_vis", action="store_true", help="Skip visualization/video output (export only)")
    parser.add_argument("--vis_mode", default="together", choices=["together", "overlay_only"],
                        help="'together' (4-panel) or 'overlay_only' (mesh on original)")
    # Smoothing args
    parser.add_argument("--no_smooth", action="store_true", help="Disable temporal smoothing")
    parser.add_argument("--smooth_window", type=int, default=11, help="Savgol window length (odd)")
    parser.add_argument("--smooth_poly", type=int, default=3, help="Savgol polynomial order")
    # Output scaling
    parser.add_argument("--scale", type=float, default=0, help="Output scale factor (e.g. 0.5). 0=auto-fit to 4K limit")
    parser.add_argument("--output_dir", default="", help="Output directory for sidecar/abc exports (default: next to input)")
    args = parser.parse_args()

    # Ensure window is odd
    if args.smooth_window % 2 == 0:
        args.smooth_window += 1

    # Defaults
    if not args.output:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(os.path.dirname(args.input) or ".", f"{base}_sam3d.mp4")
    if (args.export_obj or args.export_abc) and not args.obj_dir:
        base = os.path.splitext(os.path.basename(args.input))[0]
        if args.output_dir:
            args.obj_dir = os.path.join(args.output_dir, f"{base}_obj")
        else:
            args.obj_dir = os.path.join(os.path.dirname(args.input) or ".", f"{base}_obj")

    # Load model
    print("Loading SAM 3D Body model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint, device=device, mhr_path=args.mhr_path
    )
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=model_cfg)
    faces = estimator.faces  # (36874, 3)

    # Get UV data for .obj/.abc export
    texcoords = None
    texcoord_faces = None
    if args.export_obj or args.export_abc:
        mhr = model.head_pose.mhr
        char = mhr.character_torch
        texcoords = char.mesh.texcoords.cpu().numpy()
        texcoord_faces = char.mesh.texcoord_faces.cpu().numpy()
        os.makedirs(args.obj_dir, exist_ok=True)
        if args.export_obj:
            print(f"OBJ export enabled → {args.obj_dir}/")
        if args.export_abc:
            print(f"ABC export enabled → {args.obj_dir}/")

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {args.input} ({width}x{height} @ {fps:.2f}fps, {total_frames} frames)")

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    # ===== PASS 1: Inference on all frames =====
    print(f"\n=== Pass 1: Inference ({total_frames} frames) ===")
    all_frames = []
    all_outputs = []
    temp_path = os.path.join(os.path.dirname(args.output) or ".", "_temp_frame.jpg")

    for frame_idx in tqdm(range(total_frames), desc="Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        all_frames.append(frame)

        # Estimator expects a file path
        cv2.imwrite(temp_path, frame)
        with torch.no_grad():
            outputs = estimator.process_one_image(temp_path, bbox_thr=args.bbox_thresh)
        all_outputs.append(outputs)

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    n_valid = sum(1 for o in all_outputs if len(o) > 0)
    print(f"  Detected person in {n_valid}/{len(all_outputs)} frames")

    # Focal length is constant: sqrt(w² + h²) for fixed-resolution video
    focal_length = float(np.sqrt(width**2 + height**2))

    # ===== SIDECAR EXPORT (raw, pre-smoothing) =====
    if args.export_sidecar:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = args.output_dir if args.output_dir else (os.path.dirname(args.input) or ".")
        os.makedirs(out_dir, exist_ok=True)
        sidecar_path = os.path.join(out_dir, f"{base}_sam3d_sidecar.json")
        print(f"\n=== Sidecar Export (raw predictions) ===")
        export_sidecar(sidecar_path, all_outputs, fps, width, height, focal_length)

    # ===== SMOOTHING =====
    if not args.no_smooth and n_valid >= 3:
        print(f"\n=== Temporal Smoothing (savgol window={args.smooth_window}, poly={args.smooth_poly}) ===")
        all_outputs = apply_temporal_smoothing(
            all_outputs,
            window_length=args.smooth_window,
            poly_order=args.smooth_poly,
        )
    elif args.no_smooth:
        print("\n=== Smoothing disabled ===")
    else:
        print(f"\n=== Smoothing skipped (only {n_valid} valid frames) ===")

    # ===== PASS 2: Render + Export =====
    print(f"\n=== Pass 2: Render + Export ===")
    ffmpeg_proc = None
    abc_world_verts = []  # collect for Alembic export
    abc_template_written = False

    for frame_idx in tqdm(range(len(all_frames)), desc="Rendering"):
        frame = all_frames[frame_idx]
        outputs = all_outputs[frame_idx]

        # Export .obj / collect for .abc
        if (args.export_obj or args.export_abc) and len(outputs) > 0:
            for pid, person in enumerate(outputs):
                verts = person["pred_vertices"]
                cam_t = person["pred_cam_t"]
                world_verts = transform_verts_to_world(verts, cam_t)

                if args.export_obj:
                    obj_path = os.path.join(args.obj_dir, f"frame_{frame_idx:06d}_p{pid}.obj")
                    export_obj(obj_path, world_verts, faces, texcoords, texcoord_faces)

                # For Alembic: collect person 0 verts, write template OBJ on first valid frame
                if args.export_abc and pid == 0:
                    if not abc_template_written:
                        template_path = os.path.join(args.obj_dir, "_template.obj")
                        export_obj(template_path, world_verts, faces, texcoords, texcoord_faces)
                        abc_template_written = True
                    abc_world_verts.append(world_verts)
        elif args.export_abc:
            # No detection this frame — repeat last valid frame (or zeros)
            if abc_world_verts:
                abc_world_verts.append(abc_world_verts[-1].copy())
            else:
                abc_world_verts.append(np.zeros((faces.max() + 1, 3), dtype=np.float32))

        # Render visualization (skip if --no_vis)
        if not args.no_vis:
            if args.vis_mode == "together":
                if len(outputs) > 0:
                    vis_frame = visualize_sample_together(frame, outputs, faces)
                    vis_frame = vis_frame.astype(np.uint8)
                else:
                    vis_frame = np.concatenate([frame] * 4, axis=1)
            else:  # overlay_only
                if len(outputs) > 0:
                    renderer = Renderer(focal_length=outputs[0]["focal_length"], faces=faces)
                    vis_frame = (renderer(
                        outputs[0]["pred_vertices"],
                        outputs[0]["pred_cam_t"],
                        frame.copy(),
                        mesh_base_color=(0.65, 0.74, 0.86),
                        scene_bg_color=(1, 1, 1),
                    ) * 255).astype(np.uint8)
                else:
                    vis_frame = frame

            # Auto-scale: clamp to 4096 on any dimension (H.264 limit)
            out_h, out_w = vis_frame.shape[:2]
            scale = args.scale
            if scale == 0:
                max_dim = max(out_w, out_h)
                if max_dim > 4096:
                    scale = 4096.0 / max_dim
            if scale and scale != 1.0:
                out_w = int(out_w * scale) & ~1  # ensure even
                out_h = int(out_h * scale) & ~1
                vis_frame = cv2.resize(vis_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # Initialize ffmpeg writer on first frame
            if ffmpeg_proc is None:
                print(f"  Output: {args.output} ({out_w}x{out_h})")
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{out_w}x{out_h}",
                    "-pix_fmt", "bgr24",
                    "-r", str(fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    args.output,
                ]
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

            ffmpeg_proc.stdin.write(vis_frame.tobytes())

    if ffmpeg_proc:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()

    # Export camera JSON first (needed by Alembic export for baked camera)
    cam_json_path = None
    if args.export_obj or args.export_abc:
        base = os.path.splitext(os.path.basename(args.input))[0]
        cam_json_path = os.path.join(args.obj_dir, f"{base}_camera.json")
        export_camera_json(cam_json_path, focal_length, fps, width, height)
        print(f"  Camera data: {cam_json_path}")

    # Export Alembic
    if args.export_abc and abc_world_verts:
        base = os.path.splitext(os.path.basename(args.input))[0]
        abc_path = os.path.join(args.obj_dir, f"{base}_body.abc")
        template_path = os.path.join(args.obj_dir, "_template.obj")
        n_verts = abc_world_verts[0].shape[0]
        print(f"\n=== Alembic Export ===")
        ok = export_alembic(abc_path, template_path, abc_world_verts, fps, n_verts, camera_json_path=cam_json_path)
        if ok:
            print(f"  Alembic: {abc_path}")
        # Clean up template
        if os.path.exists(template_path):
            os.remove(template_path)

    # Export Blender import script
    if cam_json_path and (args.export_obj or args.export_abc):
        blender_script_path = os.path.join(args.obj_dir, "import_blender.py")
        with open(blender_script_path, 'w') as f:
            f.write(BLENDER_IMPORT_SCRIPT.format(
                obj_dir=args.obj_dir.replace('\\', '\\\\'),
                camera_json=cam_json_path.replace('\\', '\\\\'),
            ))
        print(f"  Blender script: {blender_script_path}")

    print(f"\nDone! Processed {len(all_frames)} frames.")
    print(f"Output video: {args.output}")
    if args.export_obj:
        print(f"OBJ files: {args.obj_dir}/")
    if args.export_abc:
        base = os.path.splitext(os.path.basename(args.input))[0]
        print(f"Alembic: {os.path.join(args.obj_dir, f'{base}_body.abc')}")
    if args.export_obj:
        print(f"\nTo import in Blender: open import_blender.py in Blender's scripting tab and run it.")
    if args.export_abc:
        print(f"To import Alembic: File > Import > Alembic in Blender/Houdini, camera from camera.json")


if __name__ == "__main__":
    main()
