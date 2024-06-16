import torch
import numpy as np
from matplotlib import pyplot as plt
import torch

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3] #total is 50
shapenet_npart = {"airplane":4, "bag": 2, "cap": 2, "car": 4, "chair":4, "earphone":3, "guitar":3, "knife":2, "lamp":4, "laptop":2, "motorbike":6, "mug": 2, "pistol":3, "rocket":3, "skateboard": 3, "table":3}
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
shapenet_categ2scalar = {"airplane":0, "bag": 1, "cap": 2, "car": 3, "chair":4, "earphone":5, "guitar":6, "knife":7, "lamp":8, "laptop":9, "motorbike":10, "mug":11, "pistol":12, "rocket":13, "skateboard":14, "table":15}


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text, is_print=True):
        if is_print:
            print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def get_colors_new():
        '''
        give scalar integer index, gives rgb color with 0-1 range
        '''
        cmap1 = plt.get_cmap(plt.cm.tab10) #it was Set2, but not enough, then change to tab20
        color_array1 = cmap1.colors
        idx_change = [0,1,2,3,4,5,6,7,8,9]
        color_array1_reordered = [color_array1[i] for i in idx_change]
        color_array = color_array1_reordered #list of tupple, each is (R,G,B)
        color_array = color_array*5
        return np.asarray(color_array)

def augment_cloud(cloud, scale=[0.9, 1.1], trans=[-0.1, 0.1], jitter_sigma=0.01, jitter_clip=0.02):
    """
    cloud: [BS, n_pts, 3]
    """
    BS, N_PTS, C = cloud.shape
    # random scale
    cloud = cloud * torch.from_numpy(np.random.uniform(low=scale[0], high=scale[1], size=[BS,1,3])).to(cloud.device).type(torch.float32)

    # random translate
    cloud = cloud + torch.from_numpy(np.random.uniform(low=trans[0], high=trans[1], size=[BS,1,3])).to(cloud.device).type(torch.float32)

    # random jitter
    cloud = cloud + torch.clip(input=jitter_sigma * torch.randn(BS, N_PTS, C).to(cloud.device).type(torch.float32), 
                               min=-1*jitter_clip,
                               max=jitter_clip) #randn is standard normal with m=0 and var=1
    return cloud
    

def rotate_cloud(cloud, degree=[0,0,0]):
    degree = np.deg2rad(np.array(degree))
    
    if degree[0] != 0:
        rx = np.array([[1, 0, 0],
                    [0, np.cos(degree[0]), -np.sin(degree[0])],
                    0, np.sin(degree[0]), np.cos((degree[0]))])
        cloud = np.dot(cloud, rx)
    
    if degree[1] != 0:
        ry = np.array([[np.cos(degree[1]), 0, np.sin(degree[1])],
                    [0, 1, 0],
                    [-np.sin(degree[1]), 0, np.cos(degree[1])]])
        cloud = np.dot(cloud, ry)

    if degree[2] != 0:
        rz = np.array([[np.cos(degree[2]), -np.sin(degree[2]), 0],
                    [np.sin(degree[2]), np.cos(degree[2]), 0],
                    [0, 0, 1]])
        cloud = np.dot(cloud, rz)
    return cloud

def norm_unit_sphere(cloud):
        """
        cloud: [n_pts, 3]
        """
        xmax = cloud[:,0].max()
        xmin = cloud[:,0].min()
        xcen = (xmax+xmin)/2.0
        ymax = cloud[:,1].max()
        ymin = cloud[:,1].min()
        ycen = (ymax+ymin)/2.0
        zmax = cloud[:,2].max()
        zmin = cloud[:,2].min()
        zcen = (zmax+zmin)/2.0
        center = np.array([xcen,ycen,zcen])
        # zero centering
        cloud = cloud - center
        # scale to unit sphere
        scaler = np.linalg.norm(cloud, axis=-1, ord=2).max()
        cloud = cloud / scaler
        return cloud


def compute_overall_iou_batchwise(pred, target, skipped_cls=[-1], category=None): 
    """
    pred: [BS, n_pts=2048] --> can be torch or np
    target: [BS, n_pts=2048]  --> --> can be torch or np
    skipped_cls: list of part_id that will be skipped
    class_choice: strs
    """
    assert not category is None, "category cannot be None!" 
    
    shape_ious = []

    if torch.is_tensor(pred):
        pred = pred.cpu().data.numpy() #[BS, n_pts]
    if torch.is_tensor(target):
        target = target.cpu().data.numpy() #[BS, n_pts]

    for shape_idx in range(pred.shape[0]):
        part_ious = []
        n_part = shapenet_npart[category.lower()]
        
        for part in range(n_part): 
            if part in skipped_cls:
                continue

            I = np.sum(np.logical_and(pred[shape_idx] == part, target[shape_idx] == part)) #scalar
            U = np.sum(np.logical_or(pred[shape_idx] == part, target[shape_idx] == part)) #scalar

            if U == 0: # follow PointCLIPv2 
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious) * 100)  # each time append an average iou across all classes of this sample (sample_level!)
       
    return shape_ious   # [batch_size]

def get_class_weight(conf_bbox, cls_bbox, n_part, opt=1):
    cls_bbox_i = cls_bbox.view(-1) #[BS*n_pts*n_bbox]
    conf_bbox_i = torch.max(conf_bbox, axis=-1)[0].view(-1) #[BS*n_pts*n_bbox]
    mask_label = conf_bbox_i > 0 #[BS*n_pts*n_bbox=32*2048*120]
    conf_bbox_i_masked = conf_bbox_i[mask_label] #[m_pts having predictions]
    cls_bbox_i_masked = cls_bbox_i[mask_label] #[m_pts having predictions]
    cls_weight_mask = cls_bbox_i_masked.unsqueeze(1) == torch.arange(np.max((n_part,2))).to(conf_bbox.device).unsqueeze(0) #[m_pts, n_part]
    cls_weight = conf_bbox_i_masked.unsqueeze(1) #[m_pts, 1]
    ## any non-zero-conf-score is counted as one
    cls_weight = (cls_weight > 0).type(torch.float) #[m_pts, 1]
    cls_weight = torch.tile(cls_weight, 
                            (1,np.max((n_part,2)))) #[m_pts, n_part]
    cls_weight = cls_weight * cls_weight_mask.type(torch.float)
    cls_weight = cls_weight.sum(0) #[n_part]

    if opt == 0: # adopting "CVPR '19 - Class-Balanced Loss Based on Effective Number of Samples" paper
        beta = 0.9999
        ## formula from the paper
        cls_weight = (1-beta)/(1-(beta**cls_weight)) 
        ## normalize the weights
        cls_weight = n_part*cls_weight/cls_weight.sum()

    elif opt == 1: # use our own weighting formula
        ## inverse weight formula
        cls_weight = torch.sum(cls_weight)/(cls_weight*n_part) 
        ## smooth the weights
        cls_weight = (torch.log10(cls_weight)/torch.log10(torch.tensor([4]).to(conf_bbox.device)))+1
        cls_weight = torch.clip(cls_weight, min=1.0, max=4)
        #using sum=[0.9295, 0.5981, 1.9600, 0.6510, 4.8508]; sum_smooth=[0.9473, 0.6293, 1.4854, 0.6904, 2.1391]; using count = [0.8564, 0.6478, 1.7920, 0.6572, 4.7806]; arm", "back", "leg", "seat", "wheel"
    else:
        raise Exception("Opt is not valid!")

    return cls_weight