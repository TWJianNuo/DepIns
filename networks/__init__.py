from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN
from .dynamicIns_decoder import DynamicDecoder
from .movment_decoder import MovementDecoder
from .sfn_discriminator import SfnD

from .cycle_gan_model import CycleGANModel

from .discriminator_encoder import DiscriminatorEncoder
from .discriminator_decoder import DiscriminatorDecoder
def create_model(opt):
    instance = CycleGANModel(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
