"""Export the verified joint replay used for the synchronized case videos.

Each Cartesian target is updated alongside the recorded joint configuration.
This exports a prior kinematic run; it does not perform a new simulation.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import zmq
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from coppeliasim_remote import _polyline_samples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-json", type=Path, required=True)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--case-id", required=True, help="For example LIDC-IDRI-0037")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=23000)
args = parser.parse_args()
run_data = json.loads(args.run_json.read_text(encoding="utf-8"))
scene_path = args.scene.resolve()
output_path = args.output.resolve()
if not scene_path.is_file():
    parser.error("The matching saved scene is required")
if output_path.exists():
    parser.error("Output already exists; choose a new filename")
for key in ("collision_check_available", "needle_self_collision_check_available",
            "mount_collision_check_available", "collision_free",
            "needle_self_collision_free", "mount_collision_free"):
    if run_data.get(key) is not True:
        parser.error(f"Source run did not pass {key}")
for key in ("collision_frames", "needle_self_collision_frames", "mount_collision_frames"):
    if run_data.get(key):
        parser.error(f"Source run contains {key}")
if run_data.get("executed_steps", 0) != run_data.get("steps") or run_data.get("executed_steps", 0) < 100:
    parser.error("Source run must contain all execution steps (at least 100)")
if not 0 <= run_data.get("max_tip_position_error_mm", float("inf")) < 1:
    parser.error("Source run must have a maximum needle-tip error below 1 mm")
case_suffix = args.case_id.removeprefix("LIDC-IDRI-")
output_path.parent.mkdir(parents=True, exist_ok=True)
ff = imageio_ffmpeg.get_ffmpeg_exe()
c = RemoteAPIClient(args.host, args.port)
c.socket.setsockopt(zmq.RCVTIMEO, 10000)
c.socket.setsockopt(zmq.LINGER, 0)
s = c.require("sim")
original_fps=s.getIntProperty(s.handle_app,'idleFps'); s.setIntProperty(s.handle_app,'idleFps',0)
all_reports=[]
try:
 for suffix in [case_suffix]:
  d=run_data
  points,phases=_polyline_samples([np.array(d[k],float) for k in ['approach_point_robot_base_mm','entry_point_robot_base_mm','target_point_robot_base_mm']],total_steps=d['steps'],max_segment_mm=d['trajectory_sampling']['max_cartesian_step_mm'])
  remap={'initial':'approach','approach':'approach_to_entry','approach_to_entry':'insertion'}
  assert [remap[p] for p in phases]==d['phase_sequence']
  assert len(points)==len(d['joint_angle_history_rad'])==100
  s.loadScene(str(scene_path))
  base=s.getObject('/UR5'); tip=s.getObject('/UR5/ikTip'); target=s.getObject('/UR5/ikTarget'); joints=s.getObjectsInTree(base,s.object_joint_type,0)
  s.setObjectSelection([]); s.setBoolParam(s.boolparam_display_enabled,True)
  for j,q in zip(joints,d['joint_angle_history_rad'][0]):s.setJointPosition(j,float(q))
  s.setObjectPosition(target,base,(points[0]/1000).tolist())
  s.cameraFitToView(0,None,0,.85)
  camera=s.getObject('/DefaultCamera'); angle=s.getObjectFloatParam(camera,s.camerafloatparam_perspective_angle)
  sensor=s.createVisionSensor(1+2+4+128,[1920,1080,0,0],[.01,10,angle,.01,0,0,.23,.27,.33,0,0])
  s.setObjectAlias(sensor,'RecordingCapture'); s.setObjectPose(sensor,s.getObjectPose(camera,s.handle_world),s.handle_world)
  # Dummies do not appear in vision-sensor images. Render their true positions as small spheres.
  markers=[]
  for name,size,color in [('RecordingTipMarker',.012,[1,.72,.18]),('RecordingTargetMarker',.010,[.15,.85,.9])]:
   h=s.createPrimitiveShape(s.primitiveshape_spheroid,[size,size,size],0); s.setObjectAlias(h,name); s.setShapeColor(h,None,s.colorcomponent_ambient_diffuse,color); markers.append(h)
  dest=output_path
  vf='setsar=1,format=yuv420p'
  process=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-','-vf',vf,'-an','-c:v','libx264','-preset','medium','-crf','18','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
  errors=[]
  try:
   for frame,(q,point) in enumerate(zip(d['joint_angle_history_rad'],points)):
    s.setBoolParam(s.boolparam_display_enabled,False)
    s.setObjectPosition(target,base,(point/1000).tolist())
    for j,value in zip(joints,q):s.setJointPosition(j,float(value))
    desired=np.array(s.getObjectPosition(target,s.handle_world)); actual=np.array(s.getObjectPosition(tip,s.handle_world))
    error=float(np.linalg.norm(desired-actual)*1000); assert error<1,(suffix,frame,error); errors.append(error)
    for h,pos in zip(markers,[actual,desired]):s.setObjectPosition(h,s.handle_world,pos.tolist())
    s.handleVisionSensor(sensor); data,res=s.getVisionSensorImg(sensor); assert res==[1920,1080]
    rgb=np.frombuffer(data,np.uint8).reshape(1080,1920,3)[::-1].copy()
    # Equal temporal weight to each original sample: 100 x 5 frames = 16.67 seconds.
    for repeat in range(5):process.stdin.write(rgb.tobytes())
    s.setBoolParam(s.boolparam_display_enabled,True)
    if frame%25==0:print(suffix,'frame',frame,'error mm',round(error,4),flush=True)
  finally:
   process.stdin.close(); stderr=process.stderr.read().decode(errors='replace'); code=process.wait(); s.setBoolParam(s.boolparam_display_enabled,True)
  assert code==0,stderr
  s.removeObjects([sensor]+markers)
  check=subprocess.run([ff,'-v','error','-i',str(dest),'-f','null','-'],capture_output=True,text=True); assert check.returncode==0 and not check.stderr.strip(),check.stderr
  assert abs(max(errors)-d['max_tip_position_error_mm'])<.02
  report={'case_id':suffix,'updated_target_each_frame':True,'source_samples':100,'video_frames':500,'fps':30,'duration_s':500/30,'max_tip_target_error_mm':max(errors),'final_tip_target_error_mm':errors[-1],'source_run_max_error_mm':d['max_tip_position_error_mm'],'full_decode':'PASS','video':str(dest),'capture':'CoppeliaSim-rendered frames; dummy positions shown with sphere markers'}
  output_path.with_suffix('.replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); all_reports.append(report); print('PASS',report,flush=True)
finally:
 s.setBoolParam(s.boolparam_display_enabled,True); s.setIntProperty(s.handle_app,'idleFps',original_fps)

