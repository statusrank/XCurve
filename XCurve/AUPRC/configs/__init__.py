import yaml
import os
import sys
sys.path.append("./")

from easydict import EasyDict as edict

PWD = os.path.dirname(os.path.abspath(__file__))

def load_yaml(filepath):
    with open(filepath, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)

DefaultLossCfg = load_yaml(PWD + '/loss.yaml')
DefaultInatDatasetCfg = load_yaml(PWD + '/datasets/inaturalist.yaml')
DefaultSOPDatasetCfg = load_yaml(PWD + '/datasets/sop.yaml')
DefaultVehIDDatasetCfg = load_yaml(PWD + '/datasets/vehid.yaml')
