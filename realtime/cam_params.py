#!/usr/bin/env python3
"""Set and read back color camera parameters of the running camera node in one call (used by
cam_exposure.sh; a ros2 param CLI call per parameter takes seconds each).
  cam_params.py [name=value ...]   bool true/false, int otherwise; prints the exposure parameters
Exits 1 if the camera node does not answer within the timeout."""
import sys
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters

NODE, TIMEOUT = "/camera/camera", 5.0
SHOW = ["rgb_camera.enable_auto_exposure", "rgb_camera.exposure", "rgb_camera.gain"]


def call(node, cli, req):
    if not cli.wait_for_service(timeout_sec=TIMEOUT):
        sys.exit(f"camera node {NODE} not found")
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=TIMEOUT)
    if fut.result() is None:
        sys.exit(f"camera node {NODE} did not answer")
    return fut.result()


rclpy.init()
node = rclpy.create_node("cam_params")
sets = []
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    pv = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v == "true") if v in ("true", "false") \
        else ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(v))
    sets.append(Parameter(name=k, value=pv))
for p in sets:  # one at a time, in order: auto exposure off before the exposure is set
    r = call(node, node.create_client(SetParameters, f"{NODE}/set_parameters"), SetParameters.Request(parameters=[p]))
    if not r.results[0].successful:
        sys.exit(f"could not set {p.name}: {r.results[0].reason}")
r = call(node, node.create_client(GetParameters, f"{NODE}/get_parameters"), GetParameters.Request(names=SHOW))
for k, v in zip(SHOW, r.values):
    print(f"  {k}: {v.bool_value if v.type == ParameterType.PARAMETER_BOOL else v.integer_value}")
node.destroy_node()
rclpy.shutdown()
