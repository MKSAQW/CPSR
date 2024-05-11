import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.nn.functional as F
from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import megengine.functional as FF
import numpy as np
import time
#import megengine.distributed as dist
import torch.distributed as dist

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default='/home/user/New_idea/UniMatch-main/configs/coco.yaml')
parser.add_argument('--labeled-id-path', type=str, default='/home/user/New_idea/UniMatch-main/splits/coco/1_128/labeled.txt')
parser.add_argument('--unlabeled-id-path', type=str, default='/home/user/New_idea/UniMatch-main/splits/coco/1_128/unlabeled.txt')
parser.add_argument('--save-path', type=str, default='/home/user/New_idea/UniMatch-main/COCO/1_128')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def calculate_uncertain_class(cfg,pred_u_s1, conf_u_w_probability_copy1,certain_ratio_ema,criterion_u,ignore_mask_cutmixed):
    pseudo_label = conf_u_w_probability_copy1.clone()
    logits_s = pred_u_s1.clone()
    max_probs, max_idx = torch.max(pseudo_label,dim=1)
    sorted_logits_w, sorted_idx = conf_u_w_probability_copy1.topk(conf_u_w_probability_copy1.shape[1],dim=1,sorted=True)
    sorted_logits_s = logits_s.gather(dim=1,index = sorted_idx)

    mask = max_probs.ge(0.95).float()

    B, C ,h, w = logits_s.shape
    removed_class_idx = []
    uncertainty_avg = []
    shrink_loss = torch.tensor(0)
    loss_npl = torch.tensor(0)

    pred_strong = F.softmax(sorted_logits_s,dim=1)
    ans = 0
    if mask.mean().item()==1:
        return certain_ratio_ema,shrink_loss, loss_npl, 0
    else:
        for batch in range(0,B):
            pred_u_s1_bat = logits_s[batch,...]
            conf_current = max_probs[batch,...]
            ignore = ignore_mask_cutmixed[batch,...]
            if (conf_current.ge(0.95)).float().mean().item()==1: continue
            for c in range(2,C):
                sub_conf = torch.cat([sorted_logits_w[batch, :1, ...],sorted_logits_w[batch,c:, ...]],dim=0)
                
                sub_conf_key = F.softmax(sub_conf,dim=0).max(dim=0)[0]  # 获得最大值
                if (sub_conf_key.ge(0.95).float().mean().item()==1) or (c == C - 1):
                    sub_logits_s = torch.cat([sorted_logits_s[batch, :1], sorted_logits_s[batch, c:]], dim=0)
                    shrink_loss_cur = criterion_u(sub_logits_s.unsqueeze(0),torch.zeros([sub_logits_s.shape[1],sub_logits_s.shape[2]]).unsqueeze(0).long().cuda())
                    # shrink_loss_cur = shrink_loss_cur*  ((sub_conf_key >= 0.95) & (ignore != 255)).unsqueeze(0)
                    shrink_loss_cur = shrink_loss_cur *  (ignore != 255).unsqueeze(0)
                    shrink_loss_cur = shrink_loss_cur.sum() / (ignore != 255).sum().item()
                    shrink_loss = shrink_loss + shrink_loss_cur
                    ans = ans + (c-1)
                    

                    f1,hhh,www = pred_strong[batch,c:].shape # C H W
                    pre_s = pred_strong[batch,c:].permute(1,2,0).contiguous().view(-1,f1)
                    loss_at = (-torch.log(1-pre_s+1e-10)).sum(axis=1).mean()
                    loss_npl = loss_npl + loss_at

                    break
    # certain_ratio_ema = mask.mean().item() if certain_ratio_ema is None else (certain_ratio_ema * 0.999 + mask.mean().item() * (1 - 0.999))

    return certain_ratio_ema,(shrink_loss * 0.05)/B , loss_npl/B, ans/B

def reduce_tensor(tensor, mean=True):
    dist.all_reduce(tensor)
    if mean:
        return tensor / dist.get_world_size()
    return tensor                

def calculate_ema_loss(pred_u_s1,conf_u_w_probability_copy1,cfg):
    max_probs, max_idx = torch.max(conf_u_w_probability_copy1,dim=1)
    mask = FF.greater_equal(max_probs,0.95).to(torch.float32) # B H W
    select = FF.greater_equal(max_probs,0.95).to(torch.int32) # B H W
    maxk = cfg['nclass']
    batch_size, C ,h, w = pred_u_s1.shape
    target = max_idx.view(-1)
    pred_u_s1_copy = pred_u_s1.permute(0,2,3,1)
    pred_u_s1_copy = pred_u_s1_copy.contiguous().view(-1,C)  # 尺寸变为B × C
    _,pred = torch.topk(pred_u_s1_copy,maxk)
    pred = pred.permute(1,0)
    correct = FF.equal(pred,torch.broadcast_to(target.reshape(1,-1),pred.shape)).to(torch.float32)
    top_k = -1
    for k in list(np.arange(2,cfg['nclass']+1)):
        correct_k = correct[:k].reshape(-1).sum(0)
        acc_single = torch.mul(correct_k,100./(batch_size*h*w))
        acc_parallel = reduce_tensor(acc_single)
        if acc_parallel > 99.99:
            top_k = k
            break
    
    softmax_pred = F.softmax(pred_u_s1,dim=1)
    softmax_pred = softmax_pred.permute(0,2,3,1).contiguous().view(-1,C)
    pseudo_label = conf_u_w_probability_copy1.clone()
    pseudo_label = pseudo_label.permute(0,2,3,1).contiguous().view(-1,C)
    topkk = torch.topk(pseudo_label, top_k)[1] 
    mask_k = torch.ones_like(pseudo_label).to(torch.int32)
    mask_k.scatter(1, topkk, torch.zeros_like(topkk).to(torch.int32)) #  (B×h×w) × c
    mask_k_npl = torch.where((mask_k==1)&(softmax_pred>(0.95)**2), torch.zeros_like(mask_k), mask_k)
    loss_npl = (-torch.log(1-softmax_pred+1e-10) * mask_k_npl).sum(axis=1).mean() / (batch_size)
    return loss_npl


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_shark_loss = AverageMeter()
        total_npl_loss = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        certain_ratio_ema_1 = None
        certain_ratio_ema_2 = None
        avg_less_certainty = 0
        all_pixels = 0 
        all_less_certainty = 0
        all_high_certainty = 0

        accuracy_low_certainty = 0
        low_certainty_allpixels = 0
        true_low_allpixels = 0

        total_delete_count = 0

        total_loss_ckrm = 0
        total_loss_tpsm = 0
        total_loss_certain = 0 
        all_time = 0

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2,mask_real_u_w),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _,mask_real_u_w_mix)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            mask_real_u_w = mask_real_u_w.cuda()
            mask_real_u_w_mix = mask_real_u_w_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix_probability = F.softmax(pred_u_w_mix,dim=1)
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w_probability = F.softmax(pred_u_w,dim=1)  # B C H W
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            cutmix_box1_3_d = cutmix_box1.unsqueeze(1).repeat(1,cfg['nclass'],1,1)
            cutmix_box2_3_d = cutmix_box2.unsqueeze(1).repeat(1,cfg['nclass'],1,1)


            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            
            conf_u_w_mix_probability_copy1 = conf_u_w_mix_probability.clone()
            conf_u_w_mix_probability_copy2 = conf_u_w_mix_probability.clone()
            conf_u_w_probability_copy1 = conf_u_w_probability.clone()
            conf_u_w_probability_copy2 = conf_u_w_probability.clone()

            conf_u_w_probability_copy1[cutmix_box1_3_d == 1]  = conf_u_w_mix_probability_copy1[cutmix_box1_3_d == 1]
            conf_u_w_probability_copy2[cutmix_box2_3_d == 1] = conf_u_w_mix_probability_copy2[cutmix_box2_3_d == 1]

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            # Mask进行叠加
            ##############
            mask_1  = mask_real_u_w.clone()
            mask_2  = mask_real_u_w.clone()
            mask_1[cutmix_box1 == 1] = mask_real_u_w_mix[cutmix_box1 == 1]
            mask_2[cutmix_box2 == 1] = mask_real_u_w_mix[cutmix_box2 == 1]
            #############

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            # print(pred_x.size(),mask_x.size())
            loss_x = criterion_l(pred_x, mask_x)

            # true_low_allpixels = true_low_allpixels + ((conf_u_w_cutmixed1 < cfg['conf_thresh']) & (mask_u_w_cutmixed1 == mask_1) & (ignore_mask_cutmixed1 != 255)).sum().item()


            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            total_loss_certain = total_loss_certain + loss_u_s1

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            certain_ratio_ema_1, shrink_loss_1,loss_nlp_1,a = calculate_uncertain_class(cfg,pred_u_s1,conf_u_w_probability_copy1,certain_ratio_ema_1,criterion_u,ignore_mask_cutmixed=ignore_mask_cutmixed1)
            certain_ratio_ema_2, shrink_loss_2,loss_nlp_2,b = calculate_uncertain_class(cfg,pred_u_s2,conf_u_w_probability_copy2,certain_ratio_ema_2,criterion_u,ignore_mask_cutmixed=ignore_mask_cutmixed2)


            total_delete_count = total_delete_count + a 

            #loss_nlp_1 = calculate_ema_loss(pred_u_s1,conf_u_w_probability_copy1,cfg)
            #loss_nlp_2 = calculate_ema_loss(pred_u_s2,conf_u_w_probability_copy2,cfg)

            loss_npl = (loss_nlp_1 + loss_nlp_2) * 0.005/2

            
            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0  +   loss_npl   + (shrink_loss_1+shrink_loss_2)/2.0 

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_shark_loss.update((shrink_loss_1+shrink_loss_2).item()/2.0)
            total_npl_loss.update(loss_npl.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Loss_shark: {:.7f},Loss_npl: {:.7f} Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_shark_loss.avg, total_npl_loss.avg,total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
