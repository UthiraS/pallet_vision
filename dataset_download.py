from roboflow import Roboflow

ROBOFLOW_API_KEY = userdata.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=ROBOFLOW_API_KEY)

workspace = rf.workspace("ks-fsm9o")
project = workspace.project("pelvis-ap-x-ray")
version = project.version(3)
dataset = version.download("yolov11")