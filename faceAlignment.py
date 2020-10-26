"""Face Alignment by Attentional Combination Network"""
from networks import network
import torch
from test_config import cfg                    # config file
import numpy as np
import cv2
from utils.misc import heatmap2pts


def get_bounding_box3(box):
    """Generate 3-points for affine transform with bounding box expansion"""
    bounding_box3 = np.zeros((3, 3), np.float32)  # [[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    width = x2-x1
    height = y2-y1
    x1 -= width * cfg.bbox_extend_factor[0]
    x2 += width * cfg.bbox_extend_factor[0]
    y2 += height * (cfg.bbox_extend_factor[0]*2 / (1 + cfg.bbox_extend_factor[0]))

    bounding_box3[0, :] = [x1, x2, x2]  # [x1, x2, x3]
    bounding_box3[1, :] = [y1, y1, y2]  # [y1, y3, y3]
    bounding_box3[2, :] = [1, 1, 1]  # [1, 1, 1]
    return bounding_box3


def make_cropped_face(image, bboxes):
    """Generate normalized face image with face bounding boxes"""
    warped_faces = []
    M_invs = []
    Ms = []
    for box in bboxes:
        face_box3 = get_bounding_box3(box)
        face_box = face_box3.transpose()[:, :2]
        face_box_template = np.float32([[0, 0], [cfg.data_shape[1], 0], [cfg.data_shape[1], cfg.data_shape[0]]])
        M = cv2.getAffineTransform(face_box, face_box_template)
        M_inv = cv2.getAffineTransform(face_box_template, face_box)
        warped_face = cv2.warpAffine(image, M, (cfg.data_shape[1], cfg.data_shape[0]))
        warped_faces.append(warped_face)
        M_invs.append(M_inv)
        Ms.append(M)
    return warped_faces, Ms,  M_invs


class FaceAlignment:
    def __init__(self, device='cuda'):
        # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPUs
        self.device = device
        self.model = network.__dict__[cfg.model](cfg, pretrained=True)
        self.model = torch.nn.DataParallel(self.model).cuda()
        checkpoint_file = cfg.model_path
        self.model.load_state_dict(torch.load(checkpoint_file), strict=False)
        print("Loaded weight from '{}'".format(cfg.model_path))

    def face_alignment(self, image, bboxes):
        """ run face alignment """
        self.model.eval()
        with torch.no_grad():
            pts = []
            batches = [bboxes[i * cfg.BATCH_SIZE:(i + 1) * cfg.BATCH_SIZE] for i in
                       range((len(bboxes) + cfg.BATCH_SIZE - 1) // cfg.BATCH_SIZE)]  # divide faces into batches
            for i, batch in enumerate(batches):
                warped_faces, Ms,  M_invs = make_cropped_face(image, batch)         # generate normalized face image
                current_n_samples = len(warped_faces)

                # make input data for model
                inputs = np.zeros((len(warped_faces), 3, cfg.data_shape[0], cfg.data_shape[1]), np.float32)
                for j, face in enumerate(warped_faces):
                    inputs[j] = np.transpose(face, (2, 0, 1))
                inputs = torch.from_numpy(inputs).float() / 255
                input_var = torch.autograd.Variable(inputs.cuda())

                # network forwarding
                coordinate_out, _, _, _, combination_out = self.model(input_var)

                if cfg.model == 'COMBINATIONNET':                                                      # use combination
                    score_map_combination = combination_out.data.cpu().numpy()
                    if cfg.flip:                                                    # if we use flip technique
                        # make flipped input data for model
                        flip_inputs = warped_faces.clone()
                        for j, face in enumerate(warped_faces):
                            face = cv2.flip(face, 1)
                            flip_inputs[j] = np.transpose(face, (2, 0, 1))
                        flip_inputs = torch.from_numpy(inputs).float() / 255
                        flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

                        # network forwarding with flipped image
                        flip_coordinate_out, _, _, _, flip_combination_out = self.model(flip_input_var)

                        # score map averaging
                        flip_score_map_combination = flip_combination_out.data.cpu().numpy()
                        for j, fscore in enumerate(flip_score_map_combination):
                            fscore = fscore.transpose((1, 2, 0))
                            fscore = cv2.flip(fscore, 1)
                            fscore = list(fscore.transpose((2, 0, 1)))
                            for (q, w) in cfg.symmetry:
                                fscore[q], fscore[w] = fscore[w], fscore[q]
                            fscore = np.array(fscore)
                            score_map_combination[j] += fscore
                            score_map_combination[j] /= 2
                    out_scale = (cfg.data_shape[0] / cfg.output_shape[0], cfg.data_shape[1] / cfg.output_shape[1])

                    # generate points from heatmap
                    template_pts_combine = heatmap2pts(score_map_combination, out_scale)
                    current_template_pts = template_pts_combine[j].reshape((cfg.num_class, 1, 3))
                elif cfg.model == 'COORDINATENET':                                           # use coordinate regression
                    template_pts_combine = coordinate_out.data.cpu().numpy() * cfg.data_shape[0]
                    current_template_pts = template_pts_combine[j].reshape((cfg.num_class, 1, 2))

                # warp points from normalized faces to faces in the original image
                for j in range(current_n_samples):
                    current_result_pts = cv2.transform(current_template_pts[:, :, :2], M_invs[j])
                    current_result_pts = current_result_pts.reshape((cfg.num_class, 2))
                    pts.append(current_result_pts)
        return pts
