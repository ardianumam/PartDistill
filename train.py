import os
import torch
import argparse, json
from torch.utils.data import DataLoader
import torch.optim as optim
from model import DistillLoss, StudentNetworks
from data import DistilledData
from tqdm import tqdm
import numpy as np
from datetime import datetime
from util import compute_overall_iou_batchwise, IOStream, augment_cloud, get_class_weight

IS_DEBUG = False

def create_data_loader(args, category, n_part, shuffle_train, shuffle_test, backward_distill_start=False, model=None, drop_last_train=True, drop_last_test=False, IS_DEBUG=False, is_test_ony=False, io=None):
    # create data loader
    test_data = DistilledData(args=args, 
                              is_train=False, 
                              category=category, 
                              n_part=n_part, 
                              IS_DEBUG=IS_DEBUG, 
                              io=io)

    test_loader = DataLoader(test_data, 
                             batch_size=args.batch_size, 
                             shuffle=shuffle_test, 
                             num_workers=4, 
                             drop_last=drop_last_test)
    if is_test_ony:    
        return test_loader
    
    train_data = DistilledData(args=args, 
                               is_train=True, 
                               category=category,
                               n_part=n_part, 
                               model=model, 
                               backward_distill_start=backward_distill_start,
                               IS_DEBUG=IS_DEBUG, 
                               io=io)
    
    BS = len(train_data) if len(train_data) < args.batch_size else args.batch_size # to handle if the dataset has less data than args.batch_size
    train_loader = DataLoader(train_data, 
                              batch_size=BS, 
                              shuffle=shuffle_train, 
                              num_workers=4,
                              drop_last=drop_last_train)

    return train_loader, test_loader
        

def evaluate(model, dataloader, category=None):
    miou_list = []
    for data in tqdm(dataloader, desc=f"Evaluating test-set"):
        pc = data['pointcloud']
        cls_bbox = data['cls_bbox'] 
        conf_bbox = data['conf_bbox']
        pid = data['semantic_seg_label']

        if torch.cuda.is_available():
            pc, cls_bbox, conf_bbox, pid = pc.cuda(), cls_bbox.cuda(), conf_bbox.cuda(), pid.cuda()
            

        net_out = model(x=pc.permute(0,2,1)) #net_out=[BS, 2048, 5]
        pred = torch.argmax(net_out, axis=-1)

        miou = compute_overall_iou_batchwise(pred=pred,
                                             target=pid,
                                             category=category)
        miou_list = miou_list + miou 

    miou = np.around(np.mean(miou_list), decimals=2)
    
    return miou


def align(args, text_prompts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # define the checkpoint dir
    now = datetime.now() # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    ckpt_dir = os.path.join(args.ckpt_dir, f"partdistill_{args.exp_suffix}")
    os.makedirs(ckpt_dir, exist_ok=True)
    io = IOStream(os.path.join(ckpt_dir,'run.log'))
    io.cprint(f"Experiment, at {dt_string}, will be logged in: {ckpt_dir}")
    io.cprint(str(args), is_print=False)

    category_list = []
    miou_test_list = []


    for category_item in args.category:       
        io.cprint(f"\n************** Processing {category_item} category **************")

        n_part = len(text_prompts[category_item.capitalize()])

        # define the student model
        model = StudentNetworks(args=args,
                                num_classes=n_part,
                                text_prompts=text_prompts[category_item.capitalize()])
        model = model.to(device)

        # define the opt
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, args.n_epoch, eta_min=args.lr/100)
        
        # create the data loader and do snapshot reference if using distilled filter
        train_loader, test_loader = create_data_loader(args, 
                                                       category_item, 
                                                       n_part=n_part, 
                                                       shuffle_train=True, 
                                                       shuffle_test=False,
                                                       drop_last_train=True, 
                                                       drop_last_test=False,
                                                       IS_DEBUG=IS_DEBUG, 
                                                       io=io)
            
        # define the loss
        criterion = DistillLoss(n_part=n_part)

        # training loop
        best_iou_test = 0.0
        backward_distill_start = False
        backward_distill_modeon = False
        loss_mov_avg_prev = 0
        loss_mov_avg = []

        print("\n")
        for epoch in range(args.n_epoch):
            epoch = epoch + 1
            loss_epoch_current = []

            if backward_distill_start and not backward_distill_modeon:
                ## snapshot std. knowledge for backward distillation purpose
                checkpoint = torch.load(os.path.join(ckpt_dir, f"ckpt_best_test_{category_item}.pth"))
                model.load_state_dict(checkpoint['model_state_dict'])        
                train_loader, test_loader = create_data_loader(args, category_item, 
                                                               n_part=n_part, 
                                                               shuffle_train=True, 
                                                               shuffle_test=False, 
                                                               drop_last_train=True, 
                                                               drop_last_test=False,
                                                               backward_distill_start=backward_distill_start,
                                                               model=model,
                                                               IS_DEBUG=IS_DEBUG, 
                                                               io=io)
                backward_distill_modeon = True
                io.cprint("***** Start backward distillation! *****")
            
            for data in (tqdm(train_loader, desc=f"Training {category_item} epoch: {epoch}/{args.n_epoch}")):
                pc = data['pointcloud']
                cls_bbox = data['cls_bbox'] 
                conf_bbox = data['conf_bbox']
                pid = data['semantic_seg_label']
                stud_knowledge = data['stud_knowledge']

                if torch.cuda.is_available():
                    pc = pc.cuda()
                    cls_bbox = cls_bbox.cuda()
                    conf_bbox = conf_bbox.cuda()
                    pid = pid.cuda()
                    stud_knowledge = stud_knowledge.cuda()
                
                # augment the cloud input
                if args.use_aug:
                    pc = augment_cloud(pc) #[BS, n_pts=2048, 3]
          
                net_out = model(x=pc.permute(0,2,1)) #net_out=[BS, 2048, 5]    

                cls_weight = get_class_weight(conf_bbox, cls_bbox, n_part, opt=1)
                

                loss, _ = criterion(net_out,
                                    cls_bbox,
                                    conf_bbox,
                                    cls_weight=cls_weight,
                                    is_backward_dist=backward_distill_modeon,
                                    stud_kowledge=stud_knowledge.permute(0,2,1)) 
                
                loss_epoch_current.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
            
            scheduler.step()

            epoch_loss_avg = np.around(np.mean(loss_epoch_current), decimals=4)
            
            # detect if it's time to perform backward distillation
            loss_mov_avg.append(epoch_loss_avg)
            if len(loss_mov_avg) > args.n_mov_avg:
                loss_mov_avg.pop(0)
            loss_mov_avg_diff = abs(np.mean(loss_mov_avg)-loss_mov_avg_prev)
            loss_mov_avg_prev = np.mean(loss_mov_avg)
            if loss_mov_avg_diff < 0.01 and epoch >= args.n_mov_avg:
                backward_distill_start = True

            # evaluate
            iou_test = evaluate(model=model.eval(),
                                dataloader=test_loader,
                                category=category_item)
            
            # store best ckpt
            if iou_test > best_iou_test:
                ckpt_path_test = os.path.join(ckpt_dir, f"ckpt_best_test_{category_item}.pth")
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict':opt.state_dict(),
                            'loss':loss,
                            'scheduler_state_dict':scheduler.state_dict()},ckpt_path_test)
                print(f"Best TEST checkpoint {category_item} is stored in: {ckpt_path_test}")
                best_iou_test = iou_test
                best_epoch_test = epoch      

            epoch_text_out = f"Epoch {epoch}/{args.n_epoch} --> loss: {epoch_loss_avg}, curr. iou: {iou_test}, best iou: {best_iou_test}, best epoch: {best_epoch_test}"
            io.cprint(epoch_text_out)

        # store final csv file
        category_list.append(category_item)
        miou_test_list.append(best_iou_test)

    category_str = ','.join(category_list)
    miou_pclip_test_str = ','.join([str(item) for item in miou_test_list])

    io_log = IOStream(os.path.join(ckpt_dir,f'summary_result.csv'))
    io_log.cprint(category_str)
    io_log.cprint(miou_pclip_test_str)

       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', default=['chair'],  nargs='+')
    parser.add_argument('--n_pts', type=int, default=2048)
    parser.add_argument('--n_epoch', type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--use_aug', type=int, default=1, choices=[0, 1]) #1 and 0 to use or not to use data augmentation
    parser.add_argument('--normalize_cloud', type=int, default=1, choices=[0, 1]) #1 and 0 to normalize or not
    parser.add_argument('--ckpt_dir',  type=str, default="checkpoints") #location to store the output 
    parser.add_argument('--backbone_path',  type=str) #fraction of training set that will be used 
    parser.add_argument('--n_mov_avg', type=int, default=5) 
    parser.add_argument('--data_path', type=str) #path to directory of the pre-processed dataset
    parser.add_argument('--exp_suffix', default='', type=str)

    args = parser.parse_args()

    text_prompts = json.load(open("assets/shapenet-part-order_meta.json"))
    align(args, text_prompts)