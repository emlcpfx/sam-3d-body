#!/usr/bin/env python3
"""
Batch GUI for SAM 3D Body sidecar export.
Tkinter interface to queue multiple video files and run process_video.py on each.
"""

import os
import sys
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import timedelta


# Defaults
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "process_video.py")
CONDA_ENV = "tunet"

VIDEO_EXTENSIONS = (
    ".mov", ".mp4", ".avi", ".mkv", ".mxf", ".webm",
    ".MOV", ".MP4", ".AVI", ".MKV", ".MXF", ".WEBM",
)


class BatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM 3D Body — Batch Export")
        self.root.geometry("820x620")
        self.root.minsize(700, 500)

        self._worker_thread = None
        self._cancel_requested = False
        self._process = None  # current subprocess

        self._build_ui()
        self._load_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        pad = dict(padx=6, pady=3)

        # --- File list ---
        list_frame = ttk.LabelFrame(self.root, text="Input Videos")
        list_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.file_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=4)

        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        scroll.pack(side=tk.LEFT, fill=tk.Y, pady=4)
        self.file_list.config(yscrollcommand=scroll.set)

        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        ttk.Button(btn_frame, text="Add Files…", command=self._add_files).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Add Folder…", command=self._add_folder).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove_selected).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_list).pack(fill=tk.X, pady=2)

        # --- Output dir ---
        out_frame = ttk.LabelFrame(self.root, text="Output Directory")
        out_frame.pack(fill=tk.X, **pad)

        self.output_dir_var = tk.StringVar()
        ttk.Entry(out_frame, textvariable=self.output_dir_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).pack(
            side=tk.LEFT, padx=4, pady=4)

        # --- Options ---
        opt_frame = ttk.LabelFrame(self.root, text="Options")
        opt_frame.pack(fill=tk.X, **pad)

        row = ttk.Frame(opt_frame)
        row.pack(fill=tk.X, padx=4, pady=4)

        self.export_sidecar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="Sidecar JSON", variable=self.export_sidecar_var).pack(side=tk.LEFT, padx=4)

        self.export_abc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Alembic ABC", variable=self.export_abc_var).pack(side=tk.LEFT, padx=4)

        self.no_vis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="No Video Output", variable=self.no_vis_var).pack(side=tk.LEFT, padx=4)

        self.no_smooth_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="No Smoothing (raw)", variable=self.no_smooth_var).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(row, text="Max frames:").pack(side=tk.LEFT, padx=(4, 2))
        self.max_frames_var = tk.IntVar(value=0)
        spin = ttk.Spinbox(row, from_=0, to=99999, width=6,
                           textvariable=self.max_frames_var)
        spin.pack(side=tk.LEFT, padx=2)
        ttk.Label(row, text="(0=all)").pack(side=tk.LEFT)

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

        self.run_btn = ttk.Button(action_frame, text="Run Batch", command=self._run_batch)
        self.run_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.cancel_btn = ttk.Button(action_frame, text="Cancel", command=self._cancel,
                                     state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=4, pady=4)

        # file count label
        self.count_var = tk.StringVar(value="0 files")
        ttk.Label(action_frame, textvariable=self.count_var).pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------
    def _settings_path(self):
        return os.path.join(SCRIPT_DIR, ".batch_gui_settings.txt")

    def _load_settings(self):
        p = self._settings_path()
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    d = f.read().strip()
                    if d and os.path.isdir(d):
                        self.output_dir_var.set(d)
            except Exception:
                pass

    def _save_settings(self):
        try:
            with open(self._settings_path(), "w") as f:
                f.write(self.output_dir_var.get())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # File list management
    # ------------------------------------------------------------------
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("Video files", "*.mov *.mp4 *.avi *.mkv *.mxf *.webm"),
                       ("All files", "*.*")])
        for p in paths:
            if p not in self.file_list.get(0, tk.END):
                self.file_list.insert(tk.END, p)
        self._update_count()

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Select folder with videos")
        if not folder:
            return
        added = 0
        for fn in sorted(os.listdir(folder)):
            if any(fn.endswith(ext) for ext in VIDEO_EXTENSIONS):
                full = os.path.join(folder, fn)
                if full not in self.file_list.get(0, tk.END):
                    self.file_list.insert(tk.END, full)
                    added += 1
        self._update_count()
        if added == 0:
            messagebox.showinfo("No videos", f"No video files found in:\n{folder}")

    def _remove_selected(self):
        sel = list(self.file_list.curselection())
        for i in reversed(sel):
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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, msg):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_safe(self, msg):
        """Thread-safe log via root.after."""
        self.root.after(0, self._log, msg)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def _run_batch(self):
        files = list(self.file_list.get(0, tk.END))
        if not files:
            messagebox.showwarning("No files", "Add at least one video file.")
            return

        if not self.export_sidecar_var.get() and not self.export_abc_var.get():
            messagebox.showwarning("No export", "Enable at least one export format.")
            return

        output_dir = self.output_dir_var.get().strip()
        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._save_settings()
        self._cancel_requested = False
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.progress_var.set(0)

        self._worker_thread = threading.Thread(
            target=self._batch_worker, args=(files, output_dir), daemon=True)
        self._worker_thread.start()

    def _cancel(self):
        self._cancel_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._log_safe("--- Cancellation requested ---")

    def _batch_worker(self, files, output_dir):
        total = len(files)
        successes = 0
        failures = 0
        batch_start = time.time()

        for idx, filepath in enumerate(files):
            if self._cancel_requested:
                self._log_safe("Batch cancelled.")
                break

            basename = os.path.basename(filepath)
            self.root.after(0, self.status_var.set,
                            f"Processing {idx + 1}/{total}: {basename}")
            self._log_safe(f"\n=== [{idx + 1}/{total}] {basename} ===")

            cmd = [
                "conda", "run", "-n", CONDA_ENV,
                "python", PROCESS_SCRIPT,
                "--input", filepath,
            ]

            if output_dir:
                cmd += ["--output_dir", output_dir]

            if self.export_sidecar_var.get():
                cmd.append("--export_sidecar")
            if self.export_abc_var.get():
                cmd.append("--export_abc")
            if self.no_vis_var.get():
                cmd.append("--no_vis")
            if self.no_smooth_var.get():
                cmd.append("--no_smooth")

            max_f = self.max_frames_var.get()
            if max_f > 0:
                cmd += ["--max_frames", str(max_f)]

            self._log_safe(f"CMD: {' '.join(cmd)}")

            try:
                # Use CREATE_NO_WINDOW on Windows
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW

                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )

                for line in iter(self._process.stdout.readline, ""):
                    if self._cancel_requested:
                        break
                    stripped = line.rstrip()
                    if stripped:
                        # Show inference progress inline
                        if stripped.startswith("Inference:") or stripped.startswith("Rendering:"):
                            self.root.after(0, self.status_var.set,
                                            f"[{idx + 1}/{total}] {basename} — {stripped[:60]}")
                        else:
                            self._log_safe(stripped)

                self._process.stdout.close()
                rc = self._process.wait(timeout=1800)

                if self._cancel_requested:
                    break

                if rc == 0:
                    self._log_safe(f"OK: {basename}")
                    successes += 1
                else:
                    self._log_safe(f"FAILED (exit {rc}): {basename}")
                    failures += 1

            except Exception as e:
                self._log_safe(f"ERROR: {basename}: {e}")
                failures += 1

            # Update progress
            pct = (idx + 1) / total * 100
            self.root.after(0, self.progress_var.set, pct)

            # Update elapsed time
            elapsed = time.time() - batch_start
            self.root.after(0, self.time_var.set,
                            str(timedelta(seconds=int(elapsed))))

        # Done
        elapsed = time.time() - batch_start
        summary = (f"Batch complete: {successes} OK, {failures} failed, "
                   f"{total - successes - failures} skipped — "
                   f"{timedelta(seconds=int(elapsed))}")
        self._log_safe(f"\n{summary}")
        self.root.after(0, self.status_var.set, summary)
        self.root.after(0, self.progress_var.set, 100)
        self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.cancel_btn.config(state=tk.DISABLED))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def on_close(self):
        self._cancel_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = BatchApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
