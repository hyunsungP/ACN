import os.path
import sys


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    # ########################## Important setting. Check out this always.
    model_path = 'weights/ACN2.pth'
    GPUs = '0'  # '0, 1' or '1, 3, 6' or '0, 1, 2, 3, 4, 5, 6, 7' or etc...

    # ########################### path setting
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '..')

    # ########################### Model setting
    model = 'COMBINATIONNET'
    num_class = 68
    data_shape = (256, 256)                                         # height, width of normalized input
    output_shape = (64, 64)                                         # height, width of output
    channel_settings = [2048, 1024, 512, 256]                       # attention mask channel setting
    init_chan_num = 128                                             # heatmap network setting
    neck_size = 4
    growth_rate = 32
    class_num = 68
    layer_num = 16
    order = 1
    loss_num = 16

    # ########################## Learning setting
    workers = 4
    # IMAGES_PER_GPU = 8
    IMAGES_PER_GPU = 1

    # ######################### sigma setting
    gk4 = (4, 4)
    gk3 = (3, 3)
    gk2 = (2, 2)
    gk1 = (1, 1)
    gk0 = (0.5, 0.5)

    bbox_extend_factor = (0.1, 0.1)  # x, y

    # ###################### validation setting
    flip = False
    symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (17, 26), (18, 25), (19, 24),
                (20, 23), (21, 22), (31, 35), (32, 34), (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
                (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]
    val_coordinate = True
    val_heatmap = True
    val_combination = True
    val_nme_compare = 'COMBINATION'

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        num_gpus = len(self.GPUs.split(','))
        self.BATCH_SIZE = self.IMAGES_PER_GPU * num_gpus

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

    def save(self, file_path):
        """"Write Configuration values in file"""
        f = open(file_path, "w")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                f.write("{:30} {}".format(a, getattr(self, a)))
                f.write('\n')
        f.close()


cfg = Config()
add_pypath(cfg.root_dir)
