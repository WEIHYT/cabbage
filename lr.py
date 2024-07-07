import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import resnet18
import seaborn as sns
import math
import torch.nn as nn
import numpy as np
from utils.dataloaders import create_dataloader
from pathlib import Path

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = check_git_info()

def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    train_loader, dataset = create_dataloader(train_path,
                                                imgsz,
                                                batch_size // WORLD_SIZE,
                                                gs,
                                                single_cls,
                                                hyp=hyp,
                                                augment=True,
                                                cache=None if opt.cache == 'val' else opt.cache,
                                                rect=opt.rect,
                                                rank=LOCAL_RANK,
                                                workers=workers,
                                                image_weights=opt.image_weights,
                                                quad=opt.quad,
                                                prefix=colorstr('train: '),
                                                shuffle=True,
                                                seed=opt.seed)

    num_epochs = 100
    n=6
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)
    # n = int(nums/batch_size)

    #定义10分类网络
    model = resnet18(num_classes=10)

    # optimizer parameter groups 设置了个优化组：权重，偏置，其他参数
    pg0, pg1, pg2 = [], [], [] 
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    optimizer = optim.SGD(pg0, lr=0.01,momentum=0.937, nesterov=True)
    #给optimizer管理的参数组中增加新的组参数，
    #可为该组参数定制lr,momentum,weight_decay 等在finetune 中常用。
    optimizer.add_param_group({'params': pg1,'weight_decay':0.0005 })  # add pg2 (biases)
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)

    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.2) + 0.2
    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lf,  #传入一个函数或一个以函数为元素列表，作为学习率调整的策略
    )

    start_epoch=0
    scheduler.last_epoch = start_epoch - 1

    lr0,lr1,lr2, epochs = [], [], [] ,[]
    optimizer.zero_grad()
    for epoch in range(start_epoch,num_epochs):
        for i in range(n):  
            #训练的迭代次数
            ni = i + n * epoch      
            # Warmup 热身的迭代次数
            # if ni <= 1000:
            #     xi = [0, 1000]
            #     for j, x in enumerate(optimizer.param_groups):
            #         #一维线性插值
            #         x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, 0.01 * lf(epoch)])              
            #         if 'momentum' in x:
            #             x['momentum'] = np.interp(ni, xi, [0.8, 0.937])
            if ni <= nw:
                    xi = [0, nw]  # x interp
                    # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])
        
        pass  # iter and train here
        
        # Scheduler 学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  
        lr0.append(lr[0])
        lr1.append(lr[1])
        lr2.append(lr[2])
        
        #学习率更新
        scheduler.step()    
        epochs.append(epoch)

    plt.figure()
    plt.subplot(221)
    plt.plot(epochs, lr0, color="r",label='BN')
    plt.legend() 

    plt.subplot(222)
    plt.plot(epochs, lr1, color="b",label='weight')
    plt.legend()  

    plt.subplot(223)
    plt.plot(epochs, lr2,color="g",label='bias')
    plt.legend()

    plt.tight_layout()  
    plt.savefig('/root/cabbage/image_cut/lr_100.jpg')
    plt.show()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5-master/weights/yolov5s.pt', help='initial weights path')    
    parser.add_argument('--cfg', type=str, default='yolov5/models/yolov5s_leaves.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC _leaves.yaml', help='dataset.yaml path')

    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path') #hyp.scratch-low.yaml
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')  #默认是不开启的,主要是为了解决样本不平衡问题；开启后会对于上一轮训练效果不好的图片，在下一轮中增加一些权重
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', default=False, help='vary img-size +/- 50%%')  #是否启用多尺度训练，默认是不开启的
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=4, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr',  default=False, help='cosine LR scheduler')  #action='store_true'
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=1, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()