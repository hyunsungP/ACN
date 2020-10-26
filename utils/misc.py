import torch
import numpy as np
import cv2


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # H*W*C
    return img


def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def pts2heatmap(pts, sigma, xv, yv):                        # enhanced heatmap generation, same as coordinate2heatmapNet
    heatmap = np.exp(-((xv-pts[0])**2/(2.0*sigma[0]**2) + (yv-pts[1])**2/(2.0*sigma[1]**2)))
    # heatmap = heatmap * (heatmap > 1e-1)
    am = np.amax(heatmap)
    if am > 0.01:
        # heatmap /= am / 255.0
        heatmap /= am / 10.0
    return heatmap


def heatmap2pts(heatmap, scale):                 # enhanced point extraction from heatmap
    # heatmap: b x n x h x w
    # predictions: b x n x 2
    radius = 2
    res = heatmap.copy()
    n_samples, n_pts, h, w = heatmap.shape
    pts = np.zeros((n_samples, n_pts, 3), np.float32)
    for b in range(n_samples):
        for n in range(n_pts):
            all_min = np.amin(res[b, n])
            if all_min < 0.0:
                res[b, n] -= all_min
            idx = res[b, n].argmax()
            y, x = np.unravel_index(idx, (h, w))
            start_x = max(0, x - radius)
            start_y = max(0, y - radius)
            end_x = min(w-1, x + radius + 1)
            end_y = min(h-1, y + radius + 1)
            field = res[b, n, start_y:end_y, start_x:end_x]
            field_sum = field.sum().sum()
            if field_sum == 0.0:                                # all zero
                pts[b, n, 0] = w/2
                pts[b, n, 1] = h/2
                pts[b, n, 2] = 0.0
                continue
            field_sum_x = field.sum(axis=0)
            field_sum_y = field.sum(axis=1)
            field_sum_x /= field_sum
            field_sum_y /= field_sum
            field_x_e_sum = 0.
            field_y_e_sum = 0.
            for i in range(end_x - start_x):
                x_coordinate = start_x-x + i
                field_x_e = x_coordinate * field_sum_x[i]
                field_x_e_sum += field_x_e
            for i in range(end_y - start_y):
                y_coordinate = start_y-y + i
                field_y_e = y_coordinate * field_sum_y[i]
                field_y_e_sum += field_y_e
            pts[b, n, 0] = (x + 0.5 + field_x_e_sum) * scale[0]
            pts[b, n, 1] = (y + 0.5 + field_y_e_sum) * scale[1]
            pts[b, n, 2] = res[b, n, y, x]
    return pts


def draw_pts(image, pts, color=(0, 255, 0), thick=2):
    output = image.copy()
    for p in pts:
        for i in range(len(p)):
            output = cv2.circle(output, (p[i, 0], p[i, 1]), 1, color, thickness=thick)
    return output


def draw_bboxes(image, bounding_boxes, fill=0.0, thickness=3):
    # it will be returned
    output = image.copy()
    # fill with transparency
    if fill > 0.0:
        # fill inside bboxes
        img_fill = image.copy()
        for bbox in bounding_boxes:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            img_fill = cv2.rectangle(img_fill, p1, p2, (0, 255, 0), -1)
        # overlay
        cv2.addWeighted(img_fill, fill, output, 1.0 - fill, 0, output)
    # edge with thickness
    for bbox in bounding_boxes:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        green = int(bbox[4] * 255)
        output = cv2.rectangle(output, p1, p2, (255, green, 0), thickness)
    return output
