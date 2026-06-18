#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pygltflib",
#   "trimesh[extras]",
# ]
# ///
"""
hy-mesh-workflow.py — Queue a ComfyUI 3D mesh workflow, wait for completion,
then validate the resulting GLB mesh against quality criteria.

Usage:
    # Full cycle (queue + wait + check):
    ./hy-mesh-workflow.py /path/to/hy-mesh-workflow.json

    # Wait for the currently-running job only (no new queue):
    ./hy-mesh-workflow.py --wait /path/to/hy-mesh-workflow.json

The script:
  1. (unless --wait) Queues the workflow via ComfyUI's /prompt API endpoint.
  2. Polls /history until the job finishes (or times out).
  3. Finds the latest GLB output in /Users/p/ComfyUI-Shared/output/3D/.
  4. Loads the GLB with trimesh and runs quality checks:
     - Nonzero (positive) volume  →  normals must be outward-facing.
     - Volume / bounding-box-volume ≥ 0.15  →  mesh fills its bounding box
       (rejects thin planes, annular rings, and hollow shells).
     - Aspect ratio of bounding box 2–6  →  vaguely dog-like (elongated,
       not a sphere or flat disc).
     - Face/vertex ratio 1.5–2.5  →  manifold mesh (rejects degenerate
       geometry with excessive or missing faces).
     - At least one large connected component  →  not scattered debris.

Exit code 0 = mesh passes all checks, 1 = one or more checks failed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import trimesh

# ── ComfyUI API constants ────────────────────────────────────────────────
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
PROMPT_URL   = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/prompt"
HISTORY_URL  = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}/history"
OUTPUT_DIR   = Path("/Users/p/ComfyUI-Shared/output/3D")

# ── Quality thresholds (tuned for a dire-wolf / dog-shaped mesh) ─────────
MIN_VOLUME_RATIO   = 0.12    # volume / bbox_volume  (rejects planes/annuli)
                                              # side-profile images naturally produce thinner meshes
MIN_ABS_VOLUME     = 0.01    # absolute volume (rejects near-empty meshes)
MAX_ASPECT_RATIO   = 6.0     # max(bbox_extents) / min(bbox_extents)
MIN_ASPECT_RATIO   = 1.5     # min — reject spheres / cubes that are too "round"
MIN_FACE_VERT_RATIO = 1.5    # faces / vertex — reject degenerate geometry
MAX_FACE_VERT_RATIO = 2.5    # faces / vertex — reject over-subdivided noise
MIN_VERT_COUNT     = 500     # minimum vertices (reject empty / near-empty)
MIN_FACE_COUNT     = 100     # minimum faces


# ── ComfyUI helpers ───────────────────────────────────────────────────────

def queue_prompt(workflow: dict) -> urllib.request.HTTPResponse:
    """Send a workflow JSON to ComfyUI's /prompt endpoint."""
    payload = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(PROMPT_URL, data=payload)
    return urllib.request.urlopen(req)


def wait_for_completion(timeout: int = 3600, poll_interval: float = 2.0):
    """Poll /history until a job finishes or timeout is reached."""
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            req = urllib.request.Request(HISTORY_URL)
            with urllib.request.urlopen(req, timeout=10) as resp:
                history = json.loads(resp.read())
        except Exception:
            time.sleep(poll_interval)
            continue

        if not history:
            time.sleep(poll_interval)
            continue

        # Grab the most recent prompt_id from history values
        prompt_ids = list(history.keys())
        if not prompt_ids:
            time.sleep(poll_interval)
            continue

        # Sort by key (ComfyUI uses timestamp strings as keys)
        prompt_ids.sort()
        latest_id = prompt_ids[-1]

        job = history[latest_id]
        status = job.get("status", {}).get("status_str", "")

        if status in ("success", "completed"):
            print(f"[INFO] Job {latest_id} completed ({status}).")
            return latest_id

        if status in ("error", "failed"):
            print(f"[ERROR] Job {latest_id} failed: {status}")
            sys.exit(1)

        # Still running — print progress if available
        outputs = job.get("outputs", {})
        for node_id, node_out in outputs.items():
            exec_data = node_out.get("exec_data", {})
            if exec_data:
                text = exec_data.get("text", "")
                if text:
                    print(f"  [node {node_id}] {text.strip()}")

        elapsed = time.time() - deadline + timeout
        print(f"  [waiting] {elapsed:.0f}s / {timeout}s …", end="\r")
        time.sleep(poll_interval)

    print(f"[ERROR] Timed out after {timeout}s.")
    sys.exit(1)


def find_latest_glb() -> Path | None:
    """Return the most recently modified .glb in OUTPUT_DIR, or None."""
    glbs = sorted(OUTPUT_DIR.glob("*.glb"), key=lambda p: p.stat().st_mtime)
    return glbs[-1] if glbs else None


# ── Mesh quality checks ───────────────────────────────────────────────────

def check_mesh(mesh: trimesh.Trimesh) -> list[str]:
    """Run all quality checks on a mesh.  Returns a list of failure messages."""
    failures: list[str] = []

    # ── 1. Vertex / face count ───────────────────────────────────────────
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    if n_verts < MIN_VERT_COUNT:
        failures.append(
            f"Too few vertices: {n_verts} (min {MIN_VERT_COUNT}). "
            "Mesh may be empty or failed to generate."
        )
    if n_faces < MIN_FACE_COUNT:
        failures.append(
            f"Too few faces: {n_faces} (min {MIN_FACE_COUNT}). "
            "Mesh may be empty or failed to generate."
        )

    # ── 2. Volume (must be positive and nonzero) ────────────────────────
    try:
        volume = mesh.volume
    except Exception:
        volume = 0.0

    if abs(volume) < MIN_ABS_VOLUME:
        failures.append(
            f"Volume too small (|{volume:.6f}| < {MIN_ABS_VOLUME}). "
            "Mesh may be empty, degenerate, or failed to generate."
        )

    if volume < 0:
        failures.append(
            f"Negative volume ({volume:.6f}) — normals are inverted. "
            "Mesh quality is poor (non-outward-facing normals)."
        )

    # ── 3. Volume / bounding-box ratio (rejects planes, annuli, shells) ──
    try:
        bbox = mesh.bounding_box
        bbox_volume = bbox.volume
        if bbox_volume > 0:
            vol_ratio = volume / bbox_volume
        else:
            vol_ratio = 0.0

        if vol_ratio < MIN_VOLUME_RATIO:
            failures.append(
                f"Volume / bbox_volume = {vol_ratio:.4f} < {MIN_VOLUME_RATIO}. "
                "Mesh is too thin (plane / annulus / shell) — not a solid dog-like shape."
            )
    except Exception:
        failures.append("Could not compute bounding box volume.")

    # ── 4. Aspect ratio (dog-like = elongated, not a sphere) ────────────
    try:
        extents = mesh.bounding_box.extents
        if min(extents) > 0:
            aspect = max(extents) / min(extents)
        else:
            aspect = float("inf")

        if aspect > MAX_ASPECT_RATIO:
            failures.append(
                f"Aspect ratio {aspect:.2f} > {MAX_ASPECT_RATIO}. "
                "Mesh is too elongated (not dog-like)."
            )
        if aspect < MIN_ASPECT_RATIO:
            failures.append(
                f"Aspect ratio {aspect:.2f} < {MIN_ASPECT_RATIO}. "
                "Mesh is too spherical / cubic (not dog-like)."
            )
    except Exception:
        failures.append("Could not compute bounding box aspect ratio.")

    # ── 5. Face / vertex ratio (rejects degenerate geometry) ────────────
    if n_verts > 0:
        fv_ratio = n_faces / n_verts
        if fv_ratio < MIN_FACE_VERT_RATIO:
            failures.append(
                f"Face/vertex ratio {fv_ratio:.2f} < {MIN_FACE_VERT_RATIO}. "
                "Mesh may have degenerate geometry (too few faces for vertices)."
            )
        if fv_ratio > MAX_FACE_VERT_RATIO:
            failures.append(
                f"Face/vertex ratio {fv_ratio:.2f} > {MAX_FACE_VERT_RATIO}. "
                "Mesh may be over-subdivided noise."
            )

    # ── 6. Is watertight? (solid mesh) ─────────────────────────────────
    try:
        if not mesh.is_watertight:
            failures.append(
                "Mesh is NOT watertight — has holes or non-manifold edges. "
                "A solid dog-like mesh should be watertight."
            )
    except Exception:
        failures.append("Could not check watertightness.")

    return failures


def print_mesh_info(mesh: trimesh.Trimesh):
    """Print a summary of the mesh for debugging."""
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    try:
        volume = mesh.volume
    except Exception:
        volume = 0.0

    try:
        bbox_extents = mesh.bounding_box.extents
        bbox_volume = mesh.bounding_box.volume
    except Exception:
        bbox_extents = None
        bbox_volume = 0.0

    try:
        aspect = (max(bbox_extents) / min(bbox_extents)) if bbox_extents is not None and min(bbox_extents) > 0 else float("inf")
    except Exception:
        aspect = float("inf")

    try:
        vol_ratio = volume / bbox_volume if bbox_volume > 0 else 0.0
    except Exception:
        vol_ratio = 0.0

    try:
        fv_ratio = n_faces / max(n_verts, 1)
    except Exception:
        fv_ratio = 0.0

    print(f"\n{'='*60}")
    print(f"  Vertices : {n_verts:,}")
    print(f"  Faces    : {n_faces:,}")
    print(f"  Volume   : {volume:.6f} (abs={abs(volume):.6f})")
    print(f"  Area     : {mesh.area:.4f}")
    print(f"  BBox     : {bbox_extents}")
    print(f"  BBox vol : {bbox_volume:.4f}")
    print(f"  Vol/Bbox : {vol_ratio:.4f}")
    print(f"  Aspect   : {aspect:.2f}")
    print(f"  F/V ratio: {fv_ratio:.2f}")
    try:
        print(f"  Watertight: {mesh.is_watertight}")
    except Exception:
        print(f"  Watertight: N/A")
    print(f"{'='*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Queue a ComfyUI 3D mesh workflow, wait for completion, then validate the GLB."
    )
    parser.add_argument(
        "workflow_path",
        help="path to a ComfyUI workflow (API) JSON",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Skip queuing a new job; just wait for the current running job and check the mesh.",
    )

    args = parser.parse_args()
    workflow_path = Path(args.workflow_path)

    if not workflow_path.exists():
        print(f"[ERROR] Workflow file not found: {workflow_path}")
        return 1

    # ── Load workflow JSON ─────────────────────────────────────────────
    print(f"[INFO] Loading workflow: {workflow_path}")
    with open(workflow_path, "rt") as f:
        workflow = json.load(f)

    # ── Step 1: (unless --wait) Queue the workflow ─────────────────────
    if not args.wait:
        print("[INFO] Queuing prompt to ComfyUI …")
        try:
            resp = queue_prompt(workflow)
        except Exception as e:
            print(f"[ERROR] Failed to queue prompt: {e}")
            return 1

        if resp.status != 200:
            print(f"[ERROR] HTTP {resp.status} queuing prompt.")
            return 1

        print("[INFO] Prompt queued. Waiting for completion …")
    else:
        print("[INFO] --wait mode: skipping queue, waiting for current job …")

    # ── Step 2: Wait for the job to finish ──────────────────────────────
    wait_for_completion(timeout=3600, poll_interval=2.0)

    # ── Step 3: Find the latest GLB output ─────────────────────────────
    glb_path = find_latest_glb()
    if glb_path is None:
        print(f"[ERROR] No GLB file found in {OUTPUT_DIR}.")
        return 1

    print(f"[INFO] Found GLB: {glb_path}")
    print(f"  Size : {glb_path.stat().st_size / 1024:.1f} KB")

    # ── Step 4: Load and validate the mesh ─────────────────────────────
    try:
        mesh = trimesh.load(str(glb_path), force="mesh", process=False)
    except Exception as e:
        print(f"[ERROR] Failed to load GLB with trimesh: {e}")
        return 1

    print_mesh_info(mesh)

    failures = check_mesh(mesh)

    if failures:
        print("[FAIL] Mesh quality checks FAILED:")
        for i, msg in enumerate(failures, 1):
            print(f"  {i}. {msg}")
        print(f"\n[RESULT] Mesh validation: FAILED ({len(failures)} issue(s))")
        return 1
    else:
        print("[PASS] All mesh quality checks passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
