from .rand_selector import RandSelector
from .H_reg import HRegSelector
from .warprf import WarpRFSelector

methods_dict = {"rand": RandSelector, "H_reg": HRegSelector, "warprf": WarpRFSelector}