# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An example integration of MJWarp with the MuJoCo viewer."""

import logging
import pickle
import time
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np
import warp as wp
from absl import app
from absl import flags

import mujoco_warp as mjwarp

_MODEL_PATH = flags.DEFINE_string("mjcf", None, "Path to a MuJoCo MJCF file.", required=True)
_CLEAR_KERNEL_CACHE = flags.DEFINE_bool("clear_kernel_cache", False, "Clear kernel cache (to calculate full JIT time)")
_ENGINE = flags.DEFINE_enum("engine", "mjwarp", ["mjwarp", "mjc"], "Simulation engine")
_CONE = flags.DEFINE_enum("cone", "pyramidal", ["pyramidal", "elliptic"], "Friction cone type")
_LS_PARALLEL = flags.DEFINE_bool("ls_parallel", False, "Engine solver with parallel linesearch")
_VIEWER_GLOBAL_STATE = {
  "running": True,
  "step_once": False,
}
_NCONMAX = flags.DEFINE_integer("nconmax", None, "Maximum number of contacts.")
_NJMAX = flags.DEFINE_integer("njmax", None, "Maximum number of constraints.")
_BROADPHASE = flags.DEFINE_integer("broadphase", None, "Broadphase collision routine.")
_BROADPHASE_FILTER = flags.DEFINE_integer("broadphase_filter", None, "Broadphase collision filter routine.")
_KEYFRAME = flags.DEFINE_integer("keyframe", None, "Keyframe to initialize simulation.")


def key_callback(key: int) -> None:
  if key == 32:  # Space bar
    _VIEWER_GLOBAL_STATE["running"] = not _VIEWER_GLOBAL_STATE["running"]
    logging.info("RUNNING = %s", _VIEWER_GLOBAL_STATE["running"])
  elif key == 46:  # period
    _VIEWER_GLOBAL_STATE["step_once"] = True


def _load_model():
  spec = mujoco.MjSpec.from_file(_MODEL_PATH.value)
  # check if the file has any mujoco.sdf test plugins
  if any(p.plugin_name.startswith("mujoco.sdf") for p in spec.plugins):
    from mujoco_warp.test_data.collision_sdf.utils import register_sdf_plugins as register_sdf_plugins

    register_sdf_plugins(mjwarp.collision_sdf)
  return spec.compile()


def _compile_step(m, d):
  mjwarp.step(m, d)
  # double warmup to work around issues with compilation during graph capture:
  mjwarp.step(m, d)
  # capture the whole step function as a CUDA graph
  with wp.ScopedCapture() as capture:
    mjwarp.step(m, d)
  return capture.graph


def mujoco_octree_to_warp_volume(mjm, octadr, resolution=64):    
    oct_child = mjm.oct_child[8*octadr:].reshape(-1, 8)
    oct_aabb = mjm.oct_aabb[6*octadr:].reshape(-1, 6)
    oct_coeff = mjm.oct_coeff[8*octadr:].reshape(-1, 8)
    
    root_aabb = oct_aabb[0]
    center = root_aabb[:3]
    half_size = root_aabb[3:]
    vmin = center - half_size
    vmax = center + half_size
    
    x = np.linspace(vmin[0], vmax[0], resolution)
    y = np.linspace(vmin[1], vmax[1], resolution) 
    z = np.linspace(vmin[2], vmax[2], resolution)
    
    sdf_values = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    
    for i, px in enumerate(x):
        for j, py in enumerate(y):
            for k, pz in enumerate(z):
                point = np.array([px, py, pz])
                sdf_val = sample_octree_sdf(point, oct_child, oct_aabb, oct_coeff)
                sdf_values[i, j, k] = sdf_val
    
    volume = wp.Volume.load_from_numpy(sdf_values)
    
    return volume


def sample_octree_sdf(point, oct_child, oct_aabb, oct_coeff):
    eps = 1e-6
    node = 0
    
    while True:
        aabb = oct_aabb[node]
        center = aabb[:3]
        half_size = aabb[3:]
        vmin = center - half_size
        vmax = center + half_size
        
        if (point[0] + eps < vmin[0] or point[0] - eps > vmax[0] or
            point[1] + eps < vmin[1] or point[1] - eps > vmax[1] or  
            point[2] + eps < vmin[2] or point[2] - eps > vmax[2]):
            return 1.0
        
        coord = (point - vmin) / (vmax - vmin)
        
        children = oct_child[node]
        if np.all(children == -1):
            sdf = 0.0
            coeffs = oct_coeff[node]
            
            for j in range(8):
                w = ((coord[0] if (j & 1) else (1 - coord[0])) *
                     (coord[1] if (j & 2) else (1 - coord[1])) *
                     (coord[2] if (j & 4) else (1 - coord[2])))
                sdf += w * coeffs[j]
            
            return sdf
        
        x_child = 0 if coord[0] >= 0.5 else 1
        y_child = 0 if coord[1] >= 0.5 else 1
        z_child = 0 if coord[2] >= 0.5 else 1
        child_idx = 4*z_child + 2*y_child + x_child

        next_node = children[child_idx]
        if next_node == -1:
            return 1.0

        node = next_node 

def _main(argv: Sequence[str]) -> None:
  """Launches MuJoCo passive viewer fed by MJWarp."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  print(f"Loading model from: {_MODEL_PATH.value}.")
  if _MODEL_PATH.value.endswith(".mjb"):
    mjm = mujoco.MjModel.from_binary_path(_MODEL_PATH.value)
  else:
    mjm = _load_model()
  if _CONE.value == "pyramidal":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  elif _CONE.value == "elliptic":
    mjm.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  mjd = mujoco.MjData(mjm)
  if _KEYFRAME.value is not None:
    mujoco.mj_resetDataKeyframe(mjm, mjd, _KEYFRAME.value)
  mujoco.mj_forward(mjm, mjd)

  if _ENGINE.value == "mjc":
    print("Engine: MuJoCo C")
  else:  # mjwarp
    print("Engine: MuJoCo Warp")
    mjm_hash = pickle.dumps(mjm)
    m = mjwarp.put_model(mjm)
    m.opt.ls_parallel = _LS_PARALLEL.value
    if _BROADPHASE.value is not None:
      m.opt.broadphase = _BROADPHASE.value
    if _BROADPHASE_FILTER.value is not None:
      m.opt.broadphase_filter = _BROADPHASE_FILTER.value

    d = mjwarp.put_data(mjm, mjd, nconmax=_NCONMAX.value, njmax=_NJMAX.value)

    # todo: move this to io.py
    volumes = []
    for mesh_id in mjm.geom_dataid:
      if mesh_id != -1:
        octree_id = mjm.mesh_octadr[mesh_id]
        if octree_id == -1:
          volumes.append(0)
        else:
          volume = mujoco_octree_to_warp_volume(mjm, octree_id, resolution=64)
          volumes.append(volume.id)
    
    m.volumes = wp.array(data=volumes, dtype=wp.uint64)

    if _CLEAR_KERNEL_CACHE.value:
      wp.clear_kernel_cache()

    print("Compiling the model physics step...")
    start = time.time()
    graph = _compile_step(m, d)
    elapsed = time.time() - start
    print(f"Compilation took {elapsed}s.")

  viewer = mujoco.viewer.launch_passive(mjm, mjd, key_callback=key_callback)
  with viewer:
    while True:
      start = time.time()

      if _ENGINE.value == "mjc":
        mujoco.mj_step(mjm, mjd)
      else:  # mjwarp
        wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
        wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
        wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
        wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
        wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
        wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))

        hash = pickle.dumps(mjm)
        if hash != mjm_hash:
          mjm_hash = hash
          m = mjwarp.put_model(mjm)
          graph = _compile_step(m, d)

        if _VIEWER_GLOBAL_STATE["running"]:
          wp.capture_launch(graph)
          wp.synchronize()
        elif _VIEWER_GLOBAL_STATE["step_once"]:
          _VIEWER_GLOBAL_STATE["step_once"] = False
          wp.capture_launch(graph)
          wp.synchronize()

        mjwarp.get_data_into(mjd, mjm, d)

      viewer.sync()

      elapsed = time.time() - start
      if elapsed < mjm.opt.timestep:
        time.sleep(mjm.opt.timestep - elapsed)


def main():
  app.run(_main)


if __name__ == "__main__":
  main()
