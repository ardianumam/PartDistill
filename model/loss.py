import torch.nn as nn
import torch.nn.functional as F
import torch


class DistillLoss(nn.Module):
    def __init__(self, n_part):
        super(DistillLoss, self).__init__()
        self.n_part = n_part


    def forward(self, net_out, cls_bbox=None, conf_bbox=None, cls_weight=None, is_backward_dist=False, stud_kowledge=None): 
        """
        net_out: [BS, n_pts, n_part]
        pid: [BS, n_pts]
        cls_bbox = [BS, n_pts, n_bbox]
        conf_bbox = [BS, n_pts, n_bbox, n_part]
        cls_weight = [n_part]
        """
        
        BS, N_PTS, N_PART = net_out.shape
        net_out = net_out.permute(0,2,1) #[BS, n_part, n_pts=2048]
        meta_data = {}
        
        conf_bbox = torch.max(conf_bbox, axis=-1)[0] #[BS, n_pts, n_bbox, n_part] --> #[BS, n_pts, n_bbox]
        
        N_BBOX = cls_bbox.shape[-1]
        net_out2 = net_out.unsqueeze(-1) #[BS, n_part, n_pts, 1]
        net_out2 = net_out2.repeat(1,1,1,N_BBOX) #[BS, n_part, n_pts, n_bbox]
        loss_raw = F.cross_entropy(input=net_out2, #[BS, n_part, n_pts, n_bbox] --> unnormalized logit
                                   target=cls_bbox, #[BS, n_pts, n_bbox]
                                   reduction='none',
                                   weight=cls_weight) #loss = [BS, n_pts, n_bbox]
        
        if is_backward_dist:
            # NOTE: this backward distillation implementation is not the efficient one. More efficient version can be performed once in the dataloader, instead of here

            ref = stud_kowledge
            
            # prepare one hot prediction
            one_hot_pred = torch.argmax(ref, dim=1) #[BS, n_pts, 1]
            one_hot_pred = F.one_hot(one_hot_pred, num_classes=N_PART) #[BS, n_pts, n_part]
            one_hot_pred = one_hot_pred.unsqueeze(2).type(torch.bool) #[BS, n_pts, 1, n_part]
            
            # prepare conf bbox per part  
            conf_bbox_per_part_mask = conf_bbox.unsqueeze(-1) #[BS, n_pts, n_bbox, 1]
            conf_bbox_per_part_mask = conf_bbox_per_part_mask > 0 #[BS, n_pts, n_bbox, 1]
            cls_bboxcls_bbox_one_hot_mask = cls_bbox[:,0,:].unsqueeze(1) #cls_bboxcls_bbox_one_hot_mask=[BS, 1 , n_bbox]; cls_bbox=[BS,n_pts,n_bbox]
            cls_bboxcls_bbox_one_hot_mask = F.one_hot(cls_bboxcls_bbox_one_hot_mask, num_classes=N_PART).type(torch.bool) #[BS, 1, n_bbox, n_part]
            conf_bbox_per_part_mask = torch.logical_and(conf_bbox_per_part_mask, cls_bboxcls_bbox_one_hot_mask) #[BS, n_pts, n_bbox, n_part]
            
            # compute the IoU
            intersect = torch.logical_and(one_hot_pred,conf_bbox_per_part_mask) #[BS, n_pts, n_bbox, n_part]
            intersect = intersect.sum(1) #[BS, n_bbox, n_part]
            union = torch.clone(conf_bbox_per_part_mask) #[BS, n_pts, n_bbox, n_part]
            union = union.sum(1) #[BS, n_bbox, n_part]
            union = torch.clip(union, min=1) #to avoid zero-division and assigning with any non-zero will not impact the result since the corresponding numerators are also zero
            iou_all_part = intersect/union #[BS, n_bbox, n_part]
            mask = cls_bboxcls_bbox_one_hot_mask.squeeze(1) #[BS, n_bbox, n_part]
            iou_per_part = iou_all_part[mask] #[BS, n_bbox]
            iou_per_part = iou_per_part.view(BS,N_BBOX) 
            meta_data["bbox_weight"] = iou_per_part #[BS, n_bbox]
            iou_per_part = iou_per_part.unsqueeze(1) #[BS, 1, n_bbox]
            iou_per_part = iou_per_part * (conf_bbox > 0).type(torch.int) # [BS, n_pts, n_bbox] 
            
            # use the iou to weight the loss 
            loss_raw = loss_raw * iou_per_part #[BS, n_pts, n_bbox]
        else:
            loss_raw = loss_raw * conf_bbox #[BS, n_pts, n_bbox]

        loss_raw = loss_raw.sum(-1) #[BS, n_pts]
        denum2 = (conf_bbox>0).sum(-1) #[BS, n_pts]
        denum2 = torch.clamp(denum2, min=1)
        loss_raw = loss_raw / denum2 #[BS, n_pts]
        loss = loss_raw.mean()
        
        return loss, meta_data