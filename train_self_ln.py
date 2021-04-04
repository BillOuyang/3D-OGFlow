import argparse
from dataset_f3d import SceneflowDataset
from dataset_f3d_self import SceneflowDataset_self

import torch, numpy as np, glob, math, torch.utils.data
import datetime
from collections import OrderedDict


from model_selfocc import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
# from model_selfocc import multiScaleLoss
# from loss_self import self_multiScaleLoss
from pointconv_selfocc_util import index_points_gather as index_points, index_points_group, Conv1d, knn_point, square_distance

# from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger



from pathlib import Path
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
from main_utils import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ClippedStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, min_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        super(ClippedStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** (self.last_epoch // self.step_size), self.min_lr)
                for base_lr in self.base_lrs]




class ln_3dogflow(pl.LightningModule):
    def __init__(self, num_points, batch_size=3, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointConvSceneFlow()

        # self.len_train_loader = len_train_loader
        # self.len_val_loader = len_val_loader
        self.batch_size = batch_size
        self.num_points = num_points


        # self.total_flow_loss = 0
        # self.total_chamfer_loss = 0
        # self.total_reg_loss = 0
        # self.total_occ_loss = 0
        # self.total_flow_loss_self = 0
        #
        # self.occ_sum = 0
        # self.occ_total_loss = 0
        # self.epe_full=0
        # self.epe=0




    def forward(self, pos1, pos2, norm1, norm2):
        pred_flows, pred_masks_fw, idx1, idx2, pcs1, pc2 = self.model(pos1, pos2, norm1, norm2)
        return pred_flows, pred_masks_fw, idx1, idx2, pcs1, pc2

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=0.0001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

        optimizer.param_groups[0]['initial_lr'] = lr
        MIN_LR = 0.00001
        STEP_SIZE_LR = 10
        GAMMA_LR = 0.83
        scheduler = ClippedStepLR(optimizer, STEP_SIZE_LR, MIN_LR, GAMMA_LR)


        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 45, 85, 95, 115], gamma=0.5)
        return [optimizer], [scheduler]

    def multiScaleLoss(self, pred_flows, gt_flow, pred_occ_masks, gt_occ_masks, fps_idxs, alpha=[0.02, 0.04, 0.08, 0.16]):
        # num of scale
        num_scale = len(pred_flows)
        offset = len(fps_idxs) - num_scale + 1

        # generate GT list and masks
        gt_flows = [gt_flow]
        gt_masks = [gt_occ_masks]
        for i in range(1, len(fps_idxs) + 1):
            fps_idx = fps_idxs[i - 1]
            sub_gt_flow = index_points(gt_flows[-1], fps_idx)
            sub_gt_mask = index_points(gt_masks[-1], fps_idx)
            gt_flows.append(sub_gt_flow)
            gt_masks.append(sub_gt_mask)

        occ_sum = 0
        flow_loss = torch.zeros(1).cuda()
        occ_loss = torch.zeros(1).cuda()
        for i in range(num_scale):
            ## SF + OCC loss
            diff_flow = (pred_flows[i].permute(0, 2, 1) - gt_flows[i + offset])
            diff_mask = pred_occ_masks[i].permute(0, 2, 1) - gt_masks[i + offset]

            occ_loss += 1.4 * alpha[i] * torch.norm(diff_mask, dim=2).sum(dim=1).mean()
            flow_loss += alpha[i] * (
                        torch.norm(diff_flow, dim=2).sum(dim=1).mean() + torch.norm(diff_flow * gt_masks[i + offset],
                                                                                    dim=2).sum(dim=1).mean())

        pred_occ_mask = pred_occ_masks[0].permute(0, 2, 1) > 0.5
        occ_acc = torch.mean((pred_occ_mask.type(torch.float32) - gt_masks[0].type(torch.float32)) ** 2)
        occ_acc = 1.0 - occ_acc
        occ_sum += occ_acc

        return flow_loss, occ_loss, occ_sum


    def epe_non_occ(self, pred, labels, mask):
        '''
            return the non occluded EPE
        '''

        pred = pred.permute(0, 2, 1)
        labels = labels.permute(0, 2, 1)
        # mask = mask.cpu().numpy()
        err = torch.sqrt(torch.sum((pred - labels) ** 2, 2) + 1e-20)
        mask_sum = torch.sum(mask, 1)
        epe = torch.sum(err * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
        epe = torch.mean(epe)

        return epe

    def occ_chamfer(self, src_warp, tg, occ_fw, occ_bw, epochs):
        k = 1
        sq_dst = square_distance(src_warp.permute(0, 2, 1), tg.permute(0, 2, 1))
        dst1 = torch.topk(sq_dst, k=k, dim=-1, largest=False)[0]
        dst2 = torch.topk(sq_dst, k=k, dim=1, largest=False)[0]
        dst1 = dst1.mean(dim=-1)
        dst2 = dst2.mean(dim=1)

        mask_fw = occ_fw.squeeze(dim=1).detach().clone()
        mask_bw = occ_bw.squeeze(dim=1).detach().clone()

        mask_fw = mask_fw.clamp(0.001, 1.0)
        mask_bw = mask_bw.clamp(0.001, 1.0)

        occ_fw_sum = (mask_fw.sum(dim=1, keepdim=True)) / sq_dst.shape[1]
        occ_bw_sum = (mask_bw.sum(dim=1, keepdim=True)) / sq_dst.shape[1]

        # if epochs <= 20:
        #     return dst1, dst2
        # else:
        return mask_fw * dst1 / occ_fw_sum, mask_bw * dst2 / occ_bw_sum


    def flow_regularization(self, xyz, flow,occ_fw,nsample=9):

        xyz = xyz.permute(0, 2, 1)
        flow = flow.permute(0, 2, 1)
        mask_fw = occ_fw.squeeze(dim=1).detach().clone()


        sq_dst = square_distance(xyz, xyz)
        knn_idx = torch.topk(sq_dst, nsample, dim=-1, largest=False, sorted=False)[1]
        grouped_flow = index_points_group(flow, knn_idx)
        reg_loss = torch.norm(grouped_flow - flow.unsqueeze(2).contiguous(),p=1, dim=3).sum(dim=2) / (1.0 * nsample - 1.0)

        return mask_fw * reg_loss, (1.0 - mask_fw) * reg_loss





    def self_multiScaleLoss(self, pred_flows, pred_masks_fw, pred_flows_bw, pred_masks_bw, pred_flows_self,
                            pred_masks_self, gt_flow_self, gt_mask_self, pos1, pos2, fps_idxs_self, reg_factor, epochs,
                            alpha=[0.02, 0.04, 0.08, 0.16]):

        # num of scale
        num_scale = len(pred_flows)
        gt_masks_self = [gt_mask_self]
        gt_flows_self = [gt_flow_self]
        for i in range(1, len(fps_idxs_self) + 1):
            fps_idx = fps_idxs_self[i - 1]
            sub_mask_self = index_points(gt_masks_self[-1], fps_idx)
            gt_masks_self.append(sub_mask_self)
            sub_flow_self = index_points(gt_flows_self[-1], fps_idx)
            gt_flows_self.append(sub_flow_self)

        chamfer_loss_tot = torch.zeros(1).cuda()
        reg_loss_tot = torch.zeros(1).cuda()
        cyc_loss_tot = torch.zeros(1).cuda()
        occ_loss_tot = torch.zeros(1).cuda()
        flow_loss_tot = torch.zeros(1).cuda()

        num_neigbor = [9, 9, 9, 5]
        for i in range(num_scale):
            src = pos1[i]  ## B,3,N
            tg = pos2[i]
            flow = pred_flows[i]  ## B,3,N
            flow_bw = pred_flows_bw[i]
            src_warp = src + flow

            occ_fw = pred_masks_fw[i]  ## B,1,N
            occ_bw = pred_masks_bw[i]

            pred_occ = pred_masks_self[i]
            gt_occ = gt_masks_self[i]

            ## chamfer dst
            # sq_dst = square_distance(src_warp.permute(0, 2, 1), tg.permute(0, 2, 1))

            dst1, dst2 = self.occ_chamfer(src_warp, tg, occ_fw, occ_bw, epochs)
            chamfer_loss = alpha[i] * (dst1.sum(dim=1).mean() + dst2.sum(dim=1).mean())

            ## reg_loss
            reg_noc, reg_occ = self.flow_regularization(src, flow, occ_fw, nsample=num_neigbor[i])
            reg_loss = alpha[i] * (reg_factor * reg_noc + 3.0 * reg_occ).sum(dim=1).mean()



            ## occ loss self-supervision
            diff_mask = pred_occ.permute(0, 2, 1) - gt_occ
            occ_loss = alpha[i] * torch.norm(diff_mask, dim=2).sum(dim=1).mean()

            # bnloss = nn.BCELoss(reduction='none')
            # occ_loss = bnloss(pred_occ.permute(0, 2, 1),gt_occ).sum(dim=1).mean()

            ## flow loss self-supervison
            diff_flow = pred_flows_self[i].permute(0, 2, 1) - gt_flows_self[i]
            # flow_loss = 0.3*alpha[i]*torch.norm(diff_flow, dim=2).sum(dim=1).mean()
            if epochs <= 30:
                flow_loss = 0.6 * alpha[i] * (torch.norm(diff_flow, dim=2).sum(dim=1).mean() + torch.norm(diff_flow * gt_occ, dim=2).sum(dim=1).mean())
            else:
                flow_loss = alpha[i] * torch.norm(diff_flow * (1.0 - gt_occ), dim=2).sum(dim=1).mean()

            # flow_loss = alpha[i] * (torch.norm(diff_flow * (1.0-gt_occ), dim=2).sum(dim=1).mean())


            chamfer_loss_tot += chamfer_loss
            reg_loss_tot += reg_loss
            # cyc_loss_tot += cyclic_loss
            occ_loss_tot += occ_loss
            flow_loss_tot += flow_loss

        return chamfer_loss_tot, reg_loss_tot, occ_loss_tot, flow_loss_tot



    def training_step(self, train_batch, batch_idx):

        epochs = self.current_epoch

        reg_factor = 3.0
        if epochs >= 50:
            reg_factor = 1.5
        if epochs >= 70:
            reg_factor = 1.0

        pos1, pos2, norm1, norm2, flow, pos2_self, norm2_self, flow_self, mask1_self = train_batch

        pred_flows, pred_masks_fw, _, _, pcs1, _ = self(pos1, pos2, norm1, norm2)
        pred_flows_bw, pred_masks_bw, _, _, pcs2, _ = self(pos2, pos1, norm2, norm1)
        pred_flows_self, pred_masks_self, fps_idxs_self, _, _, _ = self(pos1, pos2_self, norm1, norm2_self)

        chamferloss, regloss, occloss, flowloss_self = self.self_multiScaleLoss(pred_flows, pred_masks_fw,
                                                                           pred_flows_bw, pred_masks_bw,
                                                                           pred_flows_self, pred_masks_self,
                                                                           flow_self, mask1_self, pcs1,
                                                                           pcs2, fps_idxs_self, reg_factor, epochs)

        diff_flow = (pred_flows[0].permute(0, 2, 1) - flow)
        flow_loss = torch.norm(diff_flow, dim=2).sum(dim=1).mean()

        self.log("train_flow_loss", flow_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_regloss", regloss, on_step=False, on_epoch=True, sync_dist=True)
        # self.log("train_pw_loss_tot", pw_loss_tot, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_chamfer", chamferloss, on_step=False, on_epoch=True, sync_dist=True)

        if epochs <=30:
            loss = chamferloss + regloss + occloss + flowloss_self
        else:
            loss = chamferloss + regloss + occloss

        return loss



    def validation_step(self, batch, batch_idx):
        pos1, pos2, norm1, norm2, flow, mask = batch

        pred_flows, pred_mask, fps_pc1_idxs, _, _, _ = self(pos1, pos2, norm1, norm2)
        eval_loss, occ_loss, occ_acc = self.multiScaleLoss(pred_flows, flow, pred_mask, mask, fps_pc1_idxs)
        epe_full = torch.norm((pred_flows[0].permute(0, 2, 1) - flow), dim=2).mean()
        epe = self.epe_non_occ(pred_flows[0], flow.permute(0, 2, 1), mask.squeeze(dim=-1))

        return {'eval_flow_loss': eval_loss, 'eval_occ_loss': occ_loss,'eval_occ_acc': occ_acc,'epe_full_tot': epe_full,'epe_tot': epe}

    def validation_epoch_end(self, outputs):

        eval_flow_loss = torch.stack([x['eval_flow_loss'] for x in outputs]).mean()
        occ_total_loss = torch.stack([x['eval_occ_loss'] for x in outputs]).mean()
        final_occ_acc = torch.stack([x['eval_occ_acc'] for x in outputs]).mean()
        mean_epe_full = torch.stack([x['epe_full_tot'] for x in outputs]).mean()
        mean_epe = torch.stack([x['epe_tot'] for x in outputs]).mean()
        self.log('mean_epe_full', mean_epe_full, on_step=False, on_epoch=True, sync_dist=True)
        self.log('OCC_acc', final_occ_acc,on_step=False, on_epoch=True, sync_dist=True)

        self.logger.experiment.add_scalars('eval',
                           {'epe_full': mean_epe_full, 'epe': mean_epe, 'occ_acc': final_occ_acc},
                           self.current_epoch)

    def train_dataloader(self):
        train_dataset = SceneflowDataset_self(npoints=4096, train=True, cache=None)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.hparams.batch_size,
                                                   num_workers=20,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)
        return train_loader


    def val_dataloader(self):
        val_dataset = SceneflowDataset(npoints=self.num_points, train=False, cache=None)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.hparams.batch_size,
                                                 num_workers=20,
                                                 pin_memory=True,
                                                 drop_last=True)
        return val_loader



def main(num_points, batch_size, epochs, use_multi_gpu, pretrain):

    learning_rate = 0.001
    model_name = '3DOGFlow_self'
    if use_multi_gpu is False:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # create check point file path
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-' % model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    weight_dir = file_dir.joinpath('best_weight/')
    weight_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    logger = TensorBoardLogger(os.path.join(file_dir, 'tb_logs'), name='ogsf_self')
    # print('tensorboard --logdir=./' + os.path.join(file_dir, 'tb_logs'))

    # file backup
    os.system('cp %s %s' % ('model_selfocc.py', log_dir))
    os.system('cp %s %s' % ('loss_self.py', log_dir))
    os.system('cp %s %s' % ('train_self_ln.py', log_dir))


    ## ln train
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='mean_epe_full',
                                          dirpath=checkpoints_dir,
                                          filename='3DOGFlow-{epoch:02d}-{mean_epe_full:.4f}-{OCC_acc:.4f}',
                                          save_top_k=5
                                          )

    model = ln_3dogflow(num_points=num_points, batch_size=batch_size,learning_rate=learning_rate)

    if pretrain is not None:
        model.model.load_state_dict(torch.load(pretrain))
        print("load from pretrained file")

    trainer = pl.Trainer(reload_dataloaders_every_epoch=True, gpus=2, max_epochs=epochs, accelerator="ddp", logger=logger, callbacks=[lr_monitor, checkpoint_callback])
    trainer.fit(model)

    # Save best weight at the end of the training
    best_weight_path = checkpoint_callback.best_model_path
    model = model.load_from_checkpoint(best_weight_path)
    torch.save(model.model.state_dict(), '%s/%s_EPE:%.4f.pth' % (weight_dir, '3DOGFlow_self',checkpoint_callback.best_model_score))
    print()


if __name__ =="__main__":
    # Args
    parser = argparse.ArgumentParser(description='train 3D-OGFlow.')
    parser.add_argument('--num_points', type=int, default=8192, help='number of point in the input point clouds')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size number for each of the GPU')
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--use_multi_gpu', type=str2bool, default=True, help='whether to use mult-gpu for the training')
    parser.add_argument('--pretrain', type=str,
                        default=None,
                        help='train from pretrained model')
    args = parser.parse_args()
    main(args.num_points, args.batch_size, args.epochs, args.use_multi_gpu, args.pretrain)