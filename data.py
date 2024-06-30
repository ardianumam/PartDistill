import os, math
import h5py, json
import random
import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

id2cat = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
        'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
cat2part = {'airplane': ['body','wing','tail','engine or frame'], 'bag': ['handle','body'], 'cap': ['panels or crown','visor or peak'], 
            'car': ['roof','hood','wheel or tire','body'],
            'chair': ['back','seat pad','leg','armrest'], 'earphone': ['earcup','headband','data wire'], 
            'guitar': ['head or tuners','neck','body'], 
            'knife': ['blade', 'handle'], 'lamp': ['leg or wire','lampshade'], 
            'laptop': ['keyboard','screen or monitor'], 
            'motorbike': ['gas tank','seat','wheel','handles or handlebars','light','engine or frame'], 'mug': ['handle', 'cup'], 
            'pistol': ['barrel', 'handle', 'trigger and guard'], 
            'rocket': ['body','fin','nose cone'], 'skateboard': ['wheel','deck','belt for foot'], 'table': ['desktop','leg or support','drawer']}
id2part2cat = [['body', 'airplane'], ['wing', 'airplane'], ['tail', 'airplane'], ['engine or frame', 'airplane'], ['handle', 'bag'], ['body', 'bag'], 
            ['panels or crown', 'cap'], ['visor or peak', 'cap'],
            ['roof', 'car'], ['hood', 'car'], ['wheel or tire',  'car'], ['body', 'car'],
            ['backrest or back', 'chair'], ['seat', 'chair'], ['leg or support', 'chair'], ['armrest', 'chair'], 
            ['earcup', 'earphone'], ['headband', 'earphone'], ['data wire',  'earphone'], 
            ['head or tuners', 'guitar'], ['neck', 'guitar'], ['body', 'guitar'], ['blade', 'knife'], ['handle', 'knife'], 
            ['support or tube of wire', 'lamp'], ['lampshade', 'lamp'], ['canopy', 'lamp'], ['support or tube of wire', 'lamp'], 
            ['keyboard', 'laptop'], ['screen or monitor', 'laptop'], ['gas tank', 'motorbike'], ['seat', 'motorbike'], ['wheel', 'motorbike'], 
            ['handles or handlebars', 'motorbike'], ['light', 'motorbike'], ['engine or frame', 'motorbike'], ['handle', 'mug'], ['cup or body', 'mug'], 
            ['barrel', 'pistol'], ['handle', 'pistol'], ['trigger and guard', 'pistol'], ['body', 'rocket'], ['fin', 'rocket'], ['nose cone', 'rocket'], 
            ['wheel', 'skateboard'], ['deck',  'skateboard'], ['belt for foot', 'skateboard'], 
            ['desktop', 'table'], ['leg or support', 'table'], ['drawer''table']]


def get_shapenet_distill_list(set_data='train', category='chair', data_path="", IS_DEBUG=False):
    assert set_data in ["train", "test", "val"]
   
    # get the class id
    sem2id_dict = json.load(open("assets/shapenet_categ-id-map.json"))["sem2id"]
    id_cls_item = ((str)(sem2id_dict[category.capitalize()])).zfill(8)
    
    # get set list
    set_list_selected = []
    set_list_path = f"assets/train_test_split/shuffled_{set_data}_file_list.json"
    f = open(set_list_path)
    set_list = json.load(f)
    counter = 0
    for item in set_list:
        if id_cls_item in item:
            full_path = item.replace("shape_data/", "")
            full_path = os.path.join(data_path, full_path, "extracted/preprocess.h5")
            set_list_selected.append(full_path)
            counter += 1
            if IS_DEBUG and counter >=32:
                break
    
    set_list_selected = sorted(set_list_selected) #e.g., 0: '/disk2/aumam/dataset/shapenet/shapenetcore_partanno_distill/03001627/355fa0f35b61fdd7aa74a6b5ee13e775/extracted/aggregated_v2.h5'
    return set_list_selected 


def norm_unit_sphere(points):
        """
        points: [n_pts, 3]
        """
        xmax = points[:,0].max()
        xmin = points[:,0].min()
        xcen = (xmax+xmin)/2.0
        ymax = points[:,1].max()
        ymin = points[:,1].min()
        ycen = (ymax+ymin)/2.0
        zmax = points[:,2].max()
        zmin = points[:,2].min()
        zcen = (zmax+zmin)/2.0
        center = np.array([xcen,ycen,zcen])
        # zero centering
        points = points - center
        # scale to unit sphere
        scaler = np.linalg.norm(points, axis=-1, ord=2).max()
        points = points / scaler
        return points


class DistilledData(Dataset):
    def __init__(self, args, is_train=True, category=None, n_part=None, backward_distill_start=False, model=None, IS_DEBUG=False, io=None):

        pid_start_offside = json.load(open("assets/shapenet_pid_start.json"))

        self.xyz = []
        self.conf_view = [] # for view-wise
        self.cls_view = [] # for view-wise
        self.conf_bboxwise = [] # for bbox-wise
        self.cls_bboxwise = [] # for bbox-wise
        self.semantic_seg = []
        self.img_dir = [] #[n_data]
        self.max_n_pred = 0
        self.max_n_bbox = 0
        total_sample = 0

        sample_list = get_shapenet_distill_list(set_data= 'train' if is_train else 'test',
                                                category=category, 
                                                data_path=args.data_path,
                                                IS_DEBUG=IS_DEBUG)

        set_data = "train" if is_train else "test"

        for file_path in tqdm(sample_list, desc=f"Loading {set_data}-set"):
            try:
                f = h5py.File(file_path) 
            except:
                io.cprint(f"Warning! This file in {set_data}-set is not found: {file_path}", is_print=False)
                continue
            xyz = f["xyz"][:] #[n_pts, 3]
            cls_view = f["cls_view"][:] #[n_view, n_pts, n_preds]
            if cls_view.shape[-1] == 0: #detect those shape having no bbox
                if is_train: # ignore for train loader
                    io.cprint(f"This file in {set_data}-set has no bbox detection: {file_path}", is_print=False)
                    continue
                else: # keep this sample for test loader
                    # workaround: give dummy values
                    cls_view = np.zeros((cls_view.shape[0], cls_view.shape[1], 1), dtype=np.int)
                    conf_view = np.zeros((cls_view.shape[0], cls_view.shape[1], 1, n_part), dtype=np.float)
            else:
                conf_view = f["conf_view"][:] #[n_view, n_pts, n_pred, n_part]
                    
            cls_bbox = f["cls_bbox"][:] #[1, n_bbox] 
            conf_bbox = f["conf_bbox"][:] #[n_pts, n_bbox, n_part] 
            semantic_seg = f["gt_semantic_seg"][:] #[n_pts]

            # sample to args.n_pts
            if len(semantic_seg) >= args.n_pts:
                idx = np.random.choice(np.arange(len(semantic_seg)), size=args.n_pts, replace=False)
            else:
                idx_1 = np.arange(len(semantic_seg))
                n_times = math.floor(args.n_pts / len(semantic_seg))
                if n_times > 1:
                    idx_1 = np.tile(idx_1, (n_times))
                n_remaining = args.n_pts%len(semantic_seg)
                idx_2 = np.random.choice(np.arange(len(semantic_seg)), size=n_remaining, replace=False)
                idx = np.concatenate((idx_1, idx_2), axis=0)
                random.shuffle(idx)    
            xyz = xyz[idx] #[n_pts, 3]
            conf_bbox = conf_bbox[idx] #[n_pts, n_bbox, n_part]
            cls_view =  cls_view[:, idx, :] #[n_view, n_pts, n_preds]
            conf_view = conf_view[:, idx, :] #[n_view, n_pts, n_pred, n_part]
            semantic_seg = semantic_seg[idx] #[n_pts]
            
            # adjust cls labels startin from zero
            semantic_seg = semantic_seg - pid_start_offside[category.lower()]
            
            self.semantic_seg.append(semantic_seg)
            self.xyz.append(xyz)
            self.cls_view.append(np.transpose(cls_view,(1,0,2))) #[n_view, n_pts, n_pred] --> [n_pts, n_view, n_pred]
            self.conf_view.append(np.transpose(conf_view,(1,0,2,3))) #[n_view, n_pts, n_pred, n_part] --> [n_pts, n_view, n_pred, n_part]
            self.cls_bboxwise.append(cls_bbox)
            self.conf_bboxwise.append(conf_bbox)
            img_dir = file_path.split("/extracted/")[0]
            img_dir = os.path.join(img_dir, "rendered_data")
            self.img_dir.append(img_dir)
            if cls_bbox.shape[-1] > self.max_n_bbox:
                self.max_n_bbox = cls_bbox.shape[-1]
            if cls_view.shape[-1] > self.max_n_pred:
                self.max_n_pred = cls_view.shape[-1]
            total_sample += 1

        # misc.
        self.sampled_n_pts = args.n_pts
        self.is_train = is_train
        self.normalize_cloud = args.normalize_cloud
        self.n_part = n_part
        
        # snapshot integrated student knowledge
        self.stud_knowledge = torch.zeros((len(self.xyz), self.xyz[0].shape[0], n_part)).type(torch.float) #[n_data, n_pts, n_part]
        if backward_distill_start and is_train:
            assert not model is None, "Model cannot be None!"
            self.snapshot_integrated_student_knowledge(model)

        loaded_msg = f"Just loaded! {category} category in {set_data}-set: {total_sample} shapes"
        if io is not None:
            io.cprint(loaded_msg)
        else:
            print(loaded_msg)
 
    def snapshot_integrated_student_knowledge(self, model):
        for idx, item in enumerate(tqdm(self.xyz, desc="***** snapshotting integrated student knowledge *****")):
            pc = torch.from_numpy(item).type(torch.float)
            if torch.cuda.is_available():
                pc = pc.cuda()
            pc = pc.unsqueeze(0)
            net_out = model(x=pc.permute(0,2,1)) #net_out=[BS=1, n_pts=2048, n_part]
            net_out = net_out.detach().cpu() 
            self.stud_knowledge[idx] = net_out
            
    def norm_unit_sphere(self, cloud):
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
    
 
    def __getitem__(self, item):
      
        return_dict = {}
        
        pointcloud = self.xyz[item] #[n_pts, 3]
        cls_view = self.cls_view[item] #[n_pts, n_view, n_pred] 
        conf_view = self.conf_view[item] #[n_pts, n_view, n_pred, n_part]
        cls_bbox = self.cls_bboxwise[item] #[1, n_bbox] 
        conf_bbox = self.conf_bboxwise[item] #[n_pts, n_bbox, n_part]
        img_dir = self.img_dir[item] #[n_data]
        stud_knowledge = self.stud_knowledge[item] #[n_data, n_pts, n_part] 
        semantic_seg_label = self.semantic_seg[item] #[n_pts]

        
        # shuffle pts order for train-set
        if self.is_train:
            idx = np.random.permutation(self.sampled_n_pts) 
        else:
            idx = np.arange(self.sampled_n_pts)
        idx = idx[0:self.sampled_n_pts]
        pointcloud = pointcloud[idx]
        cls_view = cls_view[idx] 
        conf_view = conf_view[idx] 
        if len(cls_bbox) == 1:
            cls_bbox = np.tile(cls_bbox, (self.sampled_n_pts, 1)) 
        else:
            cls_bbox = cls_bbox[idx]
        conf_bbox = conf_bbox[idx]
        semantic_seg_label = semantic_seg_label[idx]            
        stud_knowledge = stud_knowledge[idx]
        n_pts, n_view, n_pred = cls_view.shape

        # padding to have same n_bbox
        ## view-wise
        pad_remain = self.max_n_pred - n_pred
        if pad_remain > 0:
            zero_pad_cls = np.full((n_pts, n_view, pad_remain), fill_value=0, dtype=float)
            zero_pad_prob = np.full((n_pts, n_view, pad_remain, self.n_part), fill_value=0, dtype=float)
            cls_view = np.concatenate((cls_view, zero_pad_cls), axis=2)
            conf_view = np.concatenate((conf_view, zero_pad_prob), axis=2)
        
        ## bbox-wise
        n_remaining_pad = self.max_n_bbox - cls_bbox.shape[-1]
        remaining_pad_cls = np.zeros((self.sampled_n_pts, n_remaining_pad), dtype=float)
        remaining_pad_prob = np.zeros((self.sampled_n_pts, n_remaining_pad, self.n_part), dtype=float) #[n_pts, n_pad, n_part]
        cls_bbox = np.concatenate((cls_bbox, remaining_pad_cls), axis=1) #[1, n_bbox]
        conf_bbox = np.concatenate((conf_bbox, remaining_pad_prob), axis=1)

        ## normalize cloud
        if self.normalize_cloud:
            pointcloud = self.norm_unit_sphere(pointcloud)

        pointcloud = torch.from_numpy(pointcloud).type(torch.float)
        cls_view = torch.from_numpy(cls_view).type(torch.LongTensor) #[n_pts, n_view, n_pred]
        conf_view = torch.from_numpy(conf_view).type(torch.float) #[n_pts, n_view, n_pred, n_part]
        cls_bbox = torch.from_numpy(cls_bbox).type(torch.LongTensor) #[n_pts, n_bbox] 
        conf_bbox = torch.from_numpy(conf_bbox).type(torch.float) #[n_pts, n_bbox, n_part]
        semantic_seg_label = torch.from_numpy(semantic_seg_label).type(torch.LongTensor)
        return_dict['pointcloud'] = pointcloud 
        return_dict['cls_view'] = cls_view 
        return_dict['conf_view'] = conf_view 
        return_dict['cls_bbox'] = cls_bbox 
        return_dict['conf_bbox'] = conf_bbox 
        return_dict['semantic_seg_label'] = semantic_seg_label 
        return_dict['img_dir'] = img_dir
        return_dict['stud_knowledge'] = stud_knowledge 
        
        return return_dict
        

    def __len__(self):
        return len(self.xyz)
