#!/usr/bin/env python3
"""
Sidecar JSON → Smoothed Alembic (.abc) Converter

Loads exported sidecar JSON files, applies Savitzky-Golay smoothing to MHR
params in temporal space, runs MHR forward to get vertices, and exports .abc.

No SAM 3D Body model needed — only mhr_model.pt + sam3db_metadata.pt.
"""

import json
import os
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import timedelta
from tkinter import ttk, filedialog, messagebox

import numpy as np
import torch
from scipy.signal import savgol_filter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default model paths (relative to this script)
DEFAULT_MHR_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "exported", "mhr_model.pt"))
DEFAULT_META_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "exported", "sam3db_metadata.pt"))

# obj2abc binary from headcase
_OBJ2ABC_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "..", "..", "ML_repos", "headcase", "alembic_cpp", "bin", "obj2abc.exe"),
    os.path.join(SCRIPT_DIR, "..", "..", "ML_repos", "headcase", "bin", "obj2abc.exe"),
    os.path.join(SCRIPT_DIR, "..", "..", "ML_repos", "headcase", "alembic_cpp", "build", "Release", "obj2abc.exe"),
]
OBJ2ABC_PATH = next((os.path.normpath(p) for p in _OBJ2ABC_CANDIDATES if os.path.exists(p)), "obj2abc.exe")

SIDECAR_EXTENSIONS = (".json",)


# ---------------------------------------------------------------------------
# Savitzky-Golay smoothing (param-space, before MHR)
# ---------------------------------------------------------------------------

def _fit_window(window_length, poly_order, n_frames):
    """Clamp savgol window to fit available frames."""
    if n_frames < window_length:
        window_length = n_frames
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= poly_order:
        return 0
    return window_length


def smooth_array_savgol(data, window_length=11, poly_order=3):
    """Apply savgol_filter per-dimension across time axis.

    Args:
        data: (N, D) or (N,) array
    Returns:
        Smoothed array of same shape
    """
    n_frames = data.shape[0]
    wl = _fit_window(window_length, poly_order, n_frames)
    if wl == 0:
        return data
    if data.ndim == 1:
        return savgol_filter(data, wl, poly_order)
    smoothed = np.zeros_like(data)
    for d in range(data.shape[1]):
        smoothed[:, d] = savgol_filter(data[:, d], wl, poly_order)
    return smoothed


# ---------------------------------------------------------------------------
# World-space transform for .abc export
# ---------------------------------------------------------------------------

def transform_verts_to_world(vertices, cam_t):
    """Transform raw MHR vertices to world space for .abc export.

    In process_video.py, vertices get Y/Z-flipped twice (once in
    mhr_head.forward line 340, once in its transform_verts_to_world),
    so the flips cancel.  The net effect from raw MHR verts is just
    a camera-translation offset.  We replicate that here.
    """
    world = vertices.copy()
    world[:, 0] += cam_t[0]
    world[:, 1] -= cam_t[1]
    world[:, 2] -= cam_t[2]
    return world


# ---------------------------------------------------------------------------
# OBJ export (for template)
# ---------------------------------------------------------------------------

def export_obj(filepath, vertices, faces, texcoords=None, texcoord_faces=None):
    """Write .obj with optional UVs."""
    with open(filepath, 'w') as f:
        f.write(f"# SAM 3D Body mesh\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if texcoords is not None:
            f.write("\n")
            for uv in texcoords:
                f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")
        f.write("\n")
        if texcoords is not None and texcoord_faces is not None:
            for fi, ti in zip(faces, texcoord_faces):
                f.write(f"f {fi[0]+1}/{ti[0]+1} {fi[1]+1}/{ti[1]+1} {fi[2]+1}/{ti[2]+1}\n")
        else:
            for fi in faces:
                f.write(f"f {fi[0]+1} {fi[1]+1} {fi[2]+1}\n")


# ---------------------------------------------------------------------------
# Camera JSON export
# ---------------------------------------------------------------------------

def export_camera_json(filepath, focal_length_px, fps, img_width, img_height):
    sensor_w = 36.0
    lens_mm = focal_length_px * sensor_w / img_width
    out = {
        "format": "sam3d_camera",
        "version": 2,
        "fps": fps,
        "image_width": img_width,
        "image_height": img_height,
        "camera_model": "pinhole",
        "focal_length_px": focal_length_px,
        "sensor_width_mm": sensor_w,
        "lens_mm": lens_mm,
        "principal_point": [img_width / 2.0, img_height / 2.0],
        "note": "Static camera at origin looking -Z, Y-up.",
    }
    with open(filepath, 'w') as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Alembic export via obj2abc --stream
# ---------------------------------------------------------------------------

def export_alembic(abc_path, template_obj_path, all_world_verts, fps, n_verts,
                   camera_json_path=None):
    """Pipe binary HCAS data to obj2abc."""
    if not os.path.exists(OBJ2ABC_PATH):
        return False, f"obj2abc not found at {OBJ2ABC_PATH}"

    n_frames = len(all_world_verts)
    header = struct.pack(
        '<4sIIIfII I',
        b'HCAS', 1, n_frames, n_verts, fps, 1, 0, 0,
    )

    cmd = [OBJ2ABC_PATH, "--stream", "--template", template_obj_path, "--no-normals"]
    if camera_json_path and os.path.exists(camera_json_path):
        cmd += ["--camera-json", camera_json_path]
    cmd.append(abc_path)

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    proc.stdin.write(header)
    for verts in all_world_verts:
        proc.stdin.write(verts.astype(np.float32).tobytes())
    proc.stdin.close()
    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode('utf-8', errors='replace') if stderr else "unknown error"
        return False, f"obj2abc failed (code {proc.returncode}): {err}"

    info = stdout.decode('utf-8', errors='replace').strip() if stdout else ""
    return True, info


# ---------------------------------------------------------------------------
# Core pipeline: sidecar JSON → smoothed .abc
# ---------------------------------------------------------------------------

class SidecarToAbcPipeline:
    """Loads MHR model once, then converts sidecar JSONs to .abc files."""

    def __init__(self, mhr_path, meta_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mhr = None
        self.faces = None
        self.texcoords = None
        self.texcoord_faces = None
        self.mhr_path = mhr_path
        self.meta_path = meta_path

    def load_models(self, log_fn=print):
        """Load MHR TorchScript + metadata. Call once before processing."""
        log_fn(f"Loading MHR model from {self.mhr_path}...")
        self.mhr = torch.jit.load(self.mhr_path, map_location=self.device)
        self.mhr.eval()

        log_fn(f"Loading metadata from {self.meta_path}...")
        meta = torch.jit.load(self.meta_path, map_location='cpu')
        buffers = {name: buf for name, buf in meta.named_buffers()}

        self.faces = buffers["faces"].numpy().astype(np.int32)
        self.texcoords = buffers.get("texcoords")
        self.texcoord_faces = buffers.get("texcoord_faces")
        if self.texcoords is not None:
            self.texcoords = self.texcoords.numpy()
        if self.texcoord_faces is not None:
            self.texcoord_faces = self.texcoord_faces.numpy().astype(np.int32)

        log_fn(f"  MHR on {self.device}, faces: {self.faces.shape}, "
               f"UVs: {'yes' if self.texcoords is not None else 'no'}")

    def process_sidecar(self, json_path, output_dir, window_length=11,
                        poly_order=3, person_id="0", log_fn=print):
        """Convert one sidecar JSON → smoothed .abc + camera JSON.

        Returns (success: bool, message: str).
        """
        # --- Load JSON ---
        log_fn(f"Loading {os.path.basename(json_path)}...")
        with open(json_path, 'r') as f:
            sidecar = json.load(f)

        video_info = sidecar["video_info"]
        resolution = video_info["resolution"]
        fps = video_info["fps"]
        width = resolution["width"]
        height = resolution["height"]
        total_frames = video_info["total_frames"]

        if person_id not in sidecar["persons_data"]:
            available = list(sidecar["persons_data"].keys())
            return False, f"Person '{person_id}' not found. Available: {available}"

        person = sidecar["persons_data"][person_id]
        valid_frames = person["valid_frames"]
        n_valid = len(valid_frames)
        log_fn(f"  {n_valid}/{total_frames} valid frames, {width}x{height} @ {fps}fps")

        if n_valid == 0:
            return False, "No valid frames"

        # --- Extract param arrays ---
        mhr_p = person["mhr_params"]
        model_params = np.array(mhr_p["mhr_model_params"], dtype=np.float32)   # (N, 204)
        shape_params = np.array(mhr_p["shape_params"], dtype=np.float32)       # (N, 45)
        expr_params = np.array(mhr_p.get("expr_params", [[0.0]*72]*n_valid), dtype=np.float32)  # (N, 72)
        cam_t = np.array(person["pred_cam_t"], dtype=np.float32)               # (N, 3)

        log_fn(f"  Params: model({model_params.shape}), shape({shape_params.shape}), "
               f"cam_t({cam_t.shape})")

        # --- Savgol smoothing in param space ---
        if n_valid >= 3:
            log_fn(f"  Smoothing params (window={window_length}, poly={poly_order})...")
            model_params = smooth_array_savgol(model_params, window_length, poly_order)
            shape_params = smooth_array_savgol(shape_params, window_length, poly_order)
            expr_params = smooth_array_savgol(expr_params, window_length, poly_order)
            cam_t = smooth_array_savgol(cam_t, window_length, poly_order)
        else:
            log_fn(f"  Only {n_valid} frames — skipping smoothing")

        # --- MHR forward pass → vertices ---
        log_fn(f"  Running MHR forward ({n_valid} frames)...")
        all_world_verts = []

        with torch.no_grad():
            for i in range(n_valid):
                shape_t = torch.from_numpy(shape_params[i:i+1]).to(self.device)
                model_t = torch.from_numpy(model_params[i:i+1]).to(self.device)
                expr_t = torch.from_numpy(expr_params[i:i+1]).to(self.device)

                verts, _skel = self.mhr(shape_t, model_t, expr_t)
                verts_np = verts[0].cpu().numpy() / 100.0  # MHR outputs in cm

                world_verts = transform_verts_to_world(verts_np, cam_t[i])
                all_world_verts.append(world_verts)

        n_verts = all_world_verts[0].shape[0]
        log_fn(f"  Got {n_valid} frames × {n_verts} verts")

        # --- Fill gaps: if valid_frames is sparse, hold last valid for missing frames ---
        if n_valid < total_frames:
            log_fn(f"  Filling {total_frames - n_valid} missing frames (hold-last)...")
            full_verts = []
            valid_set = set(valid_frames)
            vi = 0  # index into all_world_verts
            last_verts = all_world_verts[0]
            for frame_idx in range(total_frames):
                if frame_idx in valid_set:
                    last_verts = all_world_verts[vi]
                    vi += 1
                full_verts.append(last_verts)
            all_world_verts = full_verts

        # --- Export ---
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(json_path))[0]
        # Strip _sam3d_sidecar suffix if present
        for suffix in ("_sam3d_sidecar", "_sidecar"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break

        # Camera JSON
        focal_length = float(np.sqrt(width**2 + height**2))
        cam_json_path = os.path.join(output_dir, f"{base}_camera.json")
        export_camera_json(cam_json_path, focal_length, fps, width, height)
        log_fn(f"  Camera: {cam_json_path}")

        # Template OBJ
        template_path = os.path.join(output_dir, f"_template_{base}.obj")
        export_obj(template_path, all_world_verts[0], self.faces,
                   self.texcoords, self.texcoord_faces)

        # Alembic
        abc_path = os.path.join(output_dir, f"{base}_body.abc")
        log_fn(f"  Exporting Alembic ({len(all_world_verts)} frames)...")
        ok, info = export_alembic(
            abc_path, template_path, all_world_verts, fps, n_verts,
            camera_json_path=cam_json_path,
        )

        # Clean up template
        if os.path.exists(template_path):
            os.remove(template_path)

        if ok:
            size_mb = os.path.getsize(abc_path) / (1024 * 1024)
            log_fn(f"  OK: {abc_path} ({size_mb:.1f} MB)")
            if info:
                log_fn(f"  {info}")
            return True, abc_path
        else:
            return False, info


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class SidecarToAbcApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sidecar → Smoothed ABC")
        self.root.geometry("860x660")
        self.root.minsize(720, 520)

        self._worker_thread = None
        self._cancel_requested = False
        self._pipeline = None

        self._build_ui()
        self._load_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = dict(padx=6, pady=3)

        # --- File list ---
        list_frame = ttk.LabelFrame(self.root, text="Sidecar JSON Files")
        list_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)

        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y, pady=4)
        self.file_list.config(yscrollcommand=scroll.set)

        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        ttk.Button(btn_frame, text="Add Files...", command=self._add_files).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Add Folder...", command=self._add_folder).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_list).pack(fill=tk.X, pady=2)

        # --- Output dir ---
        out_frame = ttk.LabelFrame(self.root, text="Output Directory (blank = next to input)")
        out_frame.pack(fill=tk.X, **pad)

        self.output_dir_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)
        ttk.Button(out_frame, text="Browse...", command=self._browse_output).pack(
            side=tk.LEFT, padx=4, pady=4)

        # --- Model paths ---
        model_frame = ttk.LabelFrame(self.root, text="Models")
        model_frame.pack(fill=tk.X, **pad)

        r1 = ttk.Frame(model_frame)
        r1.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r1, text="MHR:").pack(side=tk.LEFT)
        self.mhr_path_var = tk.StringVar(value=DEFAULT_MHR_PATH)
        ttk.Entry(r1, textvariable=self.mhr_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r1, text="...", width=3, command=lambda: self._browse_model(self.mhr_path_var)).pack(side=tk.LEFT)

        r2 = ttk.Frame(model_frame)
        r2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r2, text="Meta:").pack(side=tk.LEFT)
        self.meta_path_var = tk.StringVar(value=DEFAULT_META_PATH)
        ttk.Entry(r2, textvariable=self.meta_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r2, text="...", width=3, command=lambda: self._browse_model(self.meta_path_var)).pack(side=tk.LEFT)

        # --- Smoothing options ---
        smooth_frame = ttk.LabelFrame(self.root, text="Savitzky-Golay Smoothing")
        smooth_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(smooth_frame)
        row.pack(fill=tk.X, padx=4, pady=4)

        ttk.Label(row, text="Window:").pack(side=tk.LEFT, padx=(0, 2))
        self.window_var = tk.IntVar(value=11)
        ttk.Spinbox(row, from_=3, to=101, increment=2, width=5,
                     textvariable=self.window_var).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row, text="Poly order:").pack(side=tk.LEFT, padx=(0, 2))
        self.poly_var = tk.IntVar(value=3)
        ttk.Spinbox(row, from_=1, to=7, width=4,
                     textvariable=self.poly_var).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row, text="Person ID:").pack(side=tk.LEFT, padx=(0, 2))
        self.person_var = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.person_var, width=4).pack(side=tk.LEFT, padx=(0, 12))

        self.gpu_var = tk.BooleanVar(value=torch.cuda.is_available())
        ttk.Checkbutton(row, text="GPU", variable=self.gpu_var).pack(side=tk.LEFT, padx=4)

        # --- Progress ---
        prog_frame = ttk.Frame(self.root)
        prog_frame.pack(fill=tk.X, **pad)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(prog_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=4, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(prog_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=4)

        self.time_var = tk.StringVar()
        ttk.Label(prog_frame, textvariable=self.time_var).pack(side=tk.RIGHT, padx=4)

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD,
                                font=("Consolas", 9))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)

        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.pack(side=tk.LEFT, fill=tk.Y, pady=4)
        self.log_text.config(yscrollcommand=log_scroll.set)

        # --- Run / Cancel ---
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill=tk.X, **pad)

        self.run_btn = ttk.Button(action_frame, text="Convert", command=self._run_batch)
        self.run_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.cancel_btn = ttk.Button(action_frame, text="Cancel", command=self._cancel,
                                     state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.count_var = tk.StringVar(value="0 files")
        ttk.Label(action_frame, textvariable=self.count_var).pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _settings_path(self):
        return os.path.join(SCRIPT_DIR, ".sidecar_abc_gui_settings.json")

    def _load_settings(self):
        p = self._settings_path()
        if not os.path.exists(p):
            return
        try:
            with open(p, 'r') as f:
                s = json.load(f)
            if s.get("output_dir") and os.path.isdir(s["output_dir"]):
                self.output_dir_var.set(s["output_dir"])
            if s.get("mhr_path") and os.path.isfile(s["mhr_path"]):
                self.mhr_path_var.set(s["mhr_path"])
            if s.get("meta_path") and os.path.isfile(s["meta_path"]):
                self.meta_path_var.set(s["meta_path"])
            if "window" in s:
                self.window_var.set(s["window"])
            if "poly" in s:
                self.poly_var.set(s["poly"])
        except Exception:
            pass

    def _save_settings(self):
        try:
            s = {
                "output_dir": self.output_dir_var.get(),
                "mhr_path": self.mhr_path_var.get(),
                "meta_path": self.meta_path_var.get(),
                "window": self.window_var.get(),
                "poly": self.poly_var.get(),
            }
            with open(self._settings_path(), 'w') as f:
                json.dump(s, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select sidecar JSON files",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        for p in paths:
            if p not in self.file_list.get(0, tk.END):
                self.file_list.insert(tk.END, p)
        self._update_count()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with sidecar JSONs")
        if not folder:
            return
        added = 0
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".json") and "sidecar" in fn.lower():
                full = os.path.join(folder, fn)
                if full not in self.file_list.get(0, tk.END):
                    self.file_list.insert(tk.END, full)
                    added += 1
        self._update_count()
        if added == 0:
            messagebox.showinfo("No sidecars", f"No *sidecar*.json files found in:\n{folder}")

    def _remove_selected(self):
        for i in reversed(list(self.file_list.curselection())):
            self.file_list.delete(i)
        self._update_count()

    def _clear_list(self):
        self.file_list.delete(0, tk.END)
        self._update_count()

    def _update_count(self):
        n = self.file_list.size()
        self.count_var.set(f"{n} file{'s' if n != 1 else ''}")

    def _browse_output(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir_var.set(d)

    def _browse_model(self, var):
        p = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")])
        if p:
            var.set(p)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_safe(self, msg):
        self.root.after(0, self._log, msg)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def _run_batch(self):
        files = list(self.file_list.get(0, tk.END))
        if not files:
            messagebox.showwarning("No files", "Add at least one sidecar JSON.")
            return

        mhr_path = self.mhr_path_var.get().strip()
        meta_path = self.meta_path_var.get().strip()
        if not os.path.isfile(mhr_path):
            messagebox.showerror("Missing model", f"MHR model not found:\n{mhr_path}")
            return
        if not os.path.isfile(meta_path):
            messagebox.showerror("Missing model", f"Metadata not found:\n{meta_path}")
            return

        if not os.path.exists(OBJ2ABC_PATH):
            messagebox.showerror("Missing obj2abc",
                                 f"obj2abc not found at:\n{OBJ2ABC_PATH}\n\n"
                                 "Build it from headcase/alembic_cpp/")
            return

        # Ensure window is odd
        wl = self.window_var.get()
        if wl % 2 == 0:
            wl += 1
            self.window_var.set(wl)

        self._save_settings()
        self._cancel_requested = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)

        self._worker_thread = threading.Thread(
            target=self._batch_worker,
            args=(files, self.output_dir_var.get().strip(), mhr_path, meta_path,
                  wl, self.poly_var.get(), self.person_var.get().strip(),
                  self.gpu_var.get()),
            daemon=True,
        )
        self._worker_thread.start()

    def _cancel(self):
        self._cancel_requested = True
        self._log_safe("--- Cancellation requested ---")

    def _batch_worker(self, files, output_dir, mhr_path, meta_path,
                      window_length, poly_order, person_id, use_gpu):
        total = len(files)
        successes = 0
        failures = 0
        batch_start = time.time()

        # Load models once
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        pipeline = SidecarToAbcPipeline(mhr_path, meta_path, device=device)
        try:
            pipeline.load_models(log_fn=self._log_safe)
        except Exception as e:
            self._log_safe(f"ERROR loading models: {e}")
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.cancel_btn.config(state=tk.DISABLED))
            return

        for idx, filepath in enumerate(files):
            if self._cancel_requested:
                self._log_safe("Batch cancelled.")
                break

            basename = os.path.basename(filepath)
            self.root.after(0, self.status_var.set,
                            f"Processing {idx + 1}/{total}: {basename}")
            self._log_safe(f"\n=== [{idx + 1}/{total}] {basename} ===")

            # Determine output dir for this file
            if output_dir:
                out = output_dir
            else:
                out = os.path.dirname(filepath)

            try:
                ok, msg = pipeline.process_sidecar(
                    filepath, out,
                    window_length=window_length,
                    poly_order=poly_order,
                    person_id=person_id,
                    log_fn=self._log_safe,
                )
                if ok:
                    successes += 1
                else:
                    self._log_safe(f"  FAILED: {msg}")
                    failures += 1
            except Exception as e:
                self._log_safe(f"  ERROR: {e}")
                failures += 1

            pct = (idx + 1) / total * 100
            self.root.after(0, self.progress_var.set, pct)

            elapsed = time.time() - batch_start
            self.root.after(0, self.time_var.set, str(timedelta(seconds=int(elapsed))))

        # Done
        elapsed = time.time() - batch_start
        summary = (f"Done: {successes} OK, {failures} failed, "
                   f"{total - successes - failures} skipped — "
                   f"{timedelta(seconds=int(elapsed))}")
        self._log_safe(f"\n{summary}")
        self.root.after(0, self.status_var.set, summary)
        self.root.after(0, self.progress_var.set, 100)
        self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.cancel_btn.config(state=tk.DISABLED))

    # ------------------------------------------------------------------
    def on_close(self):
        self._cancel_requested = True
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SidecarToAbcApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
