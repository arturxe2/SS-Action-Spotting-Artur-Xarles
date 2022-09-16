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
import numpy as np
from sklearn.metrics import average_precision_score

#Define trainer
def trainerSS(train_loader,
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
        loss_training = trainSS(train_loader, model, criterion, 
                              optimizer, epoch + 1,
                              train=True)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("SSmodels", model_name), exist_ok=True)

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
def trainSS(dataloader,
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
        
        for i, (featsV, featsA, labels) in t:
                
            # measure data loading time
            data_time.update(time.time() - end)
            featsV = featsV.cuda()
            featsA = featsA.cuda()
            labels = labels.cuda()
            # compute output
            classV, classA, outputs = model(featsV, featsA)
            
            loss = criterion(classV, classA)
        
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


#Trainer for AS part of the model
def trainerAS(train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            #scheduler,
            criterion,
            patience,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")
    training_stage = 0

    best_loss = 9e99

    n_bad_epochs = 0
    for epoch in range(max_epochs):
        if n_bad_epochs >= patience:
            break

        
        best_model_path = os.path.join("ASmodels", model_name, "model.pth.tar")

        # train for one epoch
        loss_training = trainAS(train_loader, model, criterion, 
                              optimizer, epoch + 1, training_stage = training_stage,
                              train=True)

        # evaluate on validation set
        loss_validation = trainAS(
            val_loader, model, criterion, optimizer, epoch + 1, 
            training_stage = training_stage, train=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("ASmodels", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            n_bad_epochs = 0
            torch.save(state, best_model_path)
        
        else:
            n_bad_epochs += 1
        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model,
                model_name)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))


    return

#Define train for AS part
def trainAS(dataloader,
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
        
        for i, (featsV, featsA, labels) in t:
                
            # measure data loading time
            data_time.update(time.time() - end)
            featsV = featsV.cuda()
            featsA = featsA.cuda()
            labels = labels.cuda()
            # compute output
            classV, classA, outputs = model(featsV, featsA)
            
            loss = criterion(outputs)
        
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


#Define test function
def test(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (featsV, featsA, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            featsV = featsV.cuda()
            featsA = featsA.cuda()
            # labels = labels.cuda()
    
            # print(feats.shape)
            # feats=feats.unsqueeze(0)
            # print(feats.shape)
    
            # compute output
            classV, classA, outputs = model(featsV, featsA)
    
            all_labels.append(labels.detach().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            desc = f'Test (cls): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    # t.set_description()
    # print(AP)
    mAP = np.mean(AP)
    print(mAP, AP)
    # a_mAP = average_mAP(spotting_grountruth, spotting_predictions, model.framerate)
    # print("Average-mAP: ", a_mAP)

    return mAP