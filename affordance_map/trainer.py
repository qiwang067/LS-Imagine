import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, HeatmapDiceLoss, HeatmapSmoothnessLoss, AWingLoss, AdaptiveWingLoss, Loss_weighted, AWing
from torchvision import transforms
from utils import test_single_volume

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70) # max_epoch = 150
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda() # image_batch: torch.Size([24, 1, 224, 224]), label_batch: torch.Size([24, 224, 224])
            
            outputs = model(image_batch) # torch.Size([24, 9, 224, 224])
            assert 1==2
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

MC_IMAGE_MEAN = torch.tensor((0.3331, 0.3245, 0.3051), dtype=torch.float32).view(3, 1, 1).cuda()
MC_IMAGE_STD = torch.tensor((0.2439, 0.2493, 0.2873), dtype=torch.float32).view(3, 1, 1).cuda()

import matplotlib.pyplot as plt

def trainer_minecraft(args, model, snapshot_path):
    from datasets.dataset_minecraft import Minecraft_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train = Minecraft_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                 transform=transforms.Compose(
                                     [RandomGenerator(output_height=args.img_size, output_width=args.img_size)]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        np.random.seed(8 + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    
    # awing_loss = AWingLoss()
    awing_loss = AdaptiveWingLoss()
    # awing_loss = AWing()
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    dice_loss = HeatmapDiceLoss()
    smoothness_loss = HeatmapSmoothnessLoss()

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iters per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        epoch_loss_list = []
        epoch_awing_list = []
        epoch_mse_list = []
        epoch_mae_list = []
        epoch_dice_list = []
        epoch_smooth_list = []
        

        logging.info('==================== epoch %d/%d ====================' % (epoch_num, max_epoch))
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch, prompt_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['prompt']
            image_batch, label_batch, prompt_batch = image_batch.cuda().float(), label_batch.cuda().float(), prompt_batch.cuda().float()
            # print(f"image_batch: {image_batch.shape}, label_batch: {label_batch.shape}, prompt_batch: {prompt_batch.shape}")
            # image_batch: torch.Size([24, 3, 224, 224]), label_batch: torch.Size([24, 1, 224, 224]), prompt_batch: torch.Size([24, 512])

            
            outputs = model(image_batch, prompt_batch) # torch.Size([24, 1, 224, 224])

            loss_awing = awing_loss(outputs, label_batch)
            loss_mse = mse_loss(outputs, label_batch)
            loss_mae = mae_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch)
            loss_smooth = smoothness_loss(outputs)
            
            # print(f"loss_mse: {loss_mse}, loss_dice: {loss_dice}, loss_smooth: {loss_smooth}")

            loss = 1 * loss_mse # + 0.12 * loss_dice + 0.36 * loss_smooth
            # loss = 1.0 * loss_awing
            # print(f"loss: {loss}")
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_awing', loss_awing.item(), iter_num)
            writer.add_scalar('loss/loss_mse', loss_mse.item(), iter_num)
            writer.add_scalar('loss/loss_mae', loss_mae.item(), iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice.item(), iter_num)
            writer.add_scalar('loss/loss_smooth', loss_smooth.item(), iter_num)
            writer.add_scalar('loss/loss', loss.item(), iter_num)

            epoch_loss_list.append(loss.item())
            epoch_awing_list.append(loss_awing.item())
            epoch_mse_list.append(loss_mse.item())
            epoch_mae_list.append(loss_mae.item())
            epoch_dice_list.append(loss_dice.item())
            epoch_smooth_list.append(loss_smooth.item())

            if iter_num % 20 == 0:
                logging.info('iteration %d : loss : %f, loss_awing : %f, loss_mse : %f, loss_mae : %f, loss_dice : %f, loss_smooth : %f' %
                             (iter_num, loss.item(), loss_awing.item(), loss_mse.item(), loss_mae.item(), loss_dice.item(), loss_smooth.item()))
                
        
        epoch_loss_avg = np.mean(epoch_loss_list)
        epoch_awing_avg = np.mean(epoch_awing_list)
        epoch_mse_avg = np.mean(epoch_mse_list)
        epoch_mae_avg = np.mean(epoch_mae_list)
        epoch_dice_avg = np.mean(epoch_dice_list)
        epoch_smooth_avg = np.mean(epoch_smooth_list)

        logging.info(f"===== epoch: {epoch_num}, loss: {epoch_loss_avg}, awing: {epoch_awing_avg}, mse: {epoch_mse_avg}, mae: {epoch_mae_avg}, dice: {epoch_dice_avg}, smooth: {epoch_smooth_avg} =====")
        
        org_img_batch = sampled_batch['org_img']
        

        rows = torch.chunk(org_img_batch, 2, dim=0)  
        rows = [torch.cat(tuple(row), dim=-1) for row in rows] 
        image_grid = torch.cat(tuple(rows), dim=-2) 

        label_batch_np = label_batch.squeeze(1).detach().cpu().numpy()  
        outputs_np = outputs.squeeze(1).detach().cpu().numpy()

        colormap = plt.get_cmap('jet')
        label_batch_colored_np = np.array([colormap(label) for label in label_batch_np]) 
        outputs_colored_np = np.array([colormap(ops) for ops in outputs_np])

        label_batch_colored_np = label_batch_colored_np[..., :3]
        outputs_colored_np = outputs_colored_np[..., :3]

        label_batch_colored_np = (label_batch_colored_np * 255).astype(np.uint8)
        outputs_colored_np = (outputs_colored_np * 255).astype(np.uint8)

        label_batch_colored = torch.from_numpy(label_batch_colored_np).permute(0, 3, 1, 2) 
        outputs_colored = torch.from_numpy(outputs_colored_np).permute(0, 3, 1, 2)

        label_rows = torch.chunk(label_batch_colored, 2, dim=0)  
        label_rows = [torch.cat(tuple(row), dim=-1) for row in label_rows]  
        label_grid = torch.cat(tuple(label_rows), dim=-2) 

        outputs_rows = torch.chunk(outputs_colored, 2, dim=0) 
        outputs_rows = [torch.cat(tuple(row), dim=-1) for row in outputs_rows] 
        outputs_grid = torch.cat(tuple(outputs_rows), dim=-2) 

        writer.add_image('train/Image', image_grid, iter_num)
        writer.add_image('train/GT', label_grid, iter_num)
        writer.add_image('train/Output', outputs_grid, iter_num)

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
