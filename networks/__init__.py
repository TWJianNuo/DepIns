from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN
from .dynamicIns_decoder import DynamicDecoder
from .movment_decoder import MovementDecoder

from .cycle_gan_model import CycleGANModel

def create_model(opt):
    instance = CycleGANModel(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
