# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:34:41 2022

@author: arturxe
"""
import logging
import os
from SoccerNet.Evaluation.utils import AverageMeter
import time
from tqdm import tqdm
import torch

#Define trainer
def trainer(train_loader,
            model,
            optimizer,
            #scheduler,
            criterion,
            model_name,
            max_epochs=1000):

    logging.info("start training")
    training_stage = 0

    best_loss = 9e99

    n_bad_epochs = 0
    for epoch in range(max_epochs):
                
        best_model_path = os.path.join("SSmodels", model_name, "model.pth.tar")

        # train for one epoch
        loss_training = train(train_loader, model, criterion, 
                              optimizer, epoch + 1,
                              train=True)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_training < best_loss
        best_loss = min(loss_training, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1


    return

#Define train
def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()

    else:
        model.eval()

    end = time.time()
    
    #Potser al fer cuda() hi ha el problema
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=160) as t:
        
        for i, (featsV, featsA) in t:
                
            # measure data loading time
            data_time.update(time.time() - end)
            featsV = featsV.cuda()
            featsA = featsA.cuda()
            # compute output
            logits = model(featsV, featsA)
            
            loss = criterion(logits)
        
            # measure accuracy and record loss
            losses.update(loss.item(), featsV.size(0) + featsA.size(0))
        
            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
                optimizer.step()
        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg
