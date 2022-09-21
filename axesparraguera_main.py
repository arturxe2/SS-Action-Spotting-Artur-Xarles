# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:57:38 2022

@author: arturxe
"""
import torch
import numpy as np
import logging
from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import Model, Model2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from datetime import datetime
import time
from loss import CLIP_loss, NLLLoss, MaskLoss, NLLLoss_weights
from train import trainerSS, trainerAS, test, testSpotting

torch.manual_seed(1)
np.random.seed(1)


#Main code to read data, train the model and make predictions
def main(args):

    logging.info("Parameters:")
    for arg in vars(args):
        logging.info(arg.rjust(15) + " : " + str(getattr(args, arg)))

    # create dataset
    if not args.test_only:    
        if args.version == 2:
            dataset_Train = SoccerNetClips(path_baidu = args.baidu_path, 
                             path_audio = args.audio_path,  
                             features_baidu = args.features_baidu,
                             features_audio = args.features_audio, 
                             split=["train"], framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
               
            dataset_Valid = SoccerNetClips(path_baidu = args.baidu_path, 
                             path_audio = args.audio_path,  
                             features_baidu = args.features_baidu,
                             features_audio = args.features_audio, 
                             split=["valid"], framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
            
            
            dataset_Valid_metric  = SoccerNetClips(path_baidu = args.baidu_path, 
                             path_audio = args.audio_path,  
                             features_baidu = args.features_baidu,
                             features_audio = args.features_audio, 
                             split=["valid"], framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
            
    '''
    dataset_Test  = SoccerNetClipsTesting(path_baidu = args.baidu_path, 
                    path_audio = args.audio_path,
                    path_labels = args.labels_path,
                    features_baidu = args.features_baidu,
                    features_audio = args.features_audio, 
                    split=args.split_test, version=args.version, 
                    framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
    '''        
        
            
    # create model
    model = Model2(weights=args.load_weights, d=args.hidden_d, 
        chunk_size=args.chunk_size, framerate=args.framerate, model=args.model).cuda()
    logging.info(model)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    parameters_per_layer  = [p.numel() for p in model.parameters() if p.requires_grad]
    logging.info("Total number of parameters: " + str(total_params))
    
    # create dataloader
    if not args.test_only:
        
        train_loader = torch.utils.data.DataLoader(dataset_Train, 
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=args.max_num_worker, pin_memory=True)
               
        val_loader = torch.utils.data.DataLoader(dataset_Valid,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)


        val_metric_loader = torch.utils.data.DataLoader(dataset_Valid_metric,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.max_num_worker, pin_memory=True)
        

    

    # training parameters
    if not args.test_only:
        #criterion = NLLLoss()
        criterionVA = CLIP_loss()
        criterionMask = MaskLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LRSS, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=1e-5, amsgrad=True)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.LR,
        #                            momentum=0.9)


        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
        
        # start training
        trainerSS(train_loader, 
                model, optimizer, criterionVA, criterionMask,
                model_name=args.model_name,
                max_epochs=args.max_epochsSS)
        
        criterion = NLLLoss_weights()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LRAS, 
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=1e-5, amsgrad=True)
        
        trainerAS(train_loader,
                  val_loader,
                  val_metric_loader,
                  model, optimizer, criterion,
                  patience=args.patience,
                  model_name=args.model_name,
                  max_epochs=args.max_epochsAS)

    # For the best model only
    checkpoint = torch.load(os.path.join("ASmodels", args.model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    
    
    #print(asdf)

    # test on multiple splits [test/challenge]
    for split in args.split_test:
         
        dataset_Test  = SoccerNetClipsTesting(path_baidu = args.baidu_path, 
                        path_audio = args.audio_path,
                        path_labels = args.labels_path,
                        features_baidu = args.features_baidu,
                        features_audio = args.features_audio, 
                        split=split, 
                        framerate=args.framerate, chunk_size=args.chunk_size*args.framerate)
        print('Test loader')
        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
        
        
        results = testSpotting(test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold)
        if results is None:
            continue

        a_mAP = results["a_mAP"]
        a_mAP_per_class = results["a_mAP_per_class"]
        a_mAP_visible = results["a_mAP_visible"]
        a_mAP_per_class_visible = results["a_mAP_per_class_visible"]
        a_mAP_unshown = results["a_mAP_unshown"]
        a_mAP_per_class_unshown = results["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training ")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

    return

if __name__ == '__main__':


    parser = ArgumentParser(description='context aware loss function', formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--baidu_path', required=False, type=str, default='/data-local/data3-ssd/axesparraguera', help='path of baidu features')
    parser.add_argument('--features_baidu', required=False, type=str, default='baidu_soccer_embeddings_2fps.npy', help='baidu features name')
    parser.add_argument('--audio_path', required=False, type=str, default='/data-local/data3-ssd/axesparraguera', help='path of audio features')
    parser.add_argument('--features_audio', required=False, type=str, default='audio_embeddings_2fps.npy', help='audio features name')
    parser.add_argument('--labels_path', required=False, type=str, default='/data-net/datasets/SoccerNetv2/ResNET_TF2', help='path of labels')
    
    parser.add_argument('--max_epochsSS',   required=False, type=int,   default=20,     help='Maximum number of epochs for SS' )
    parser.add_argument('--max_epochsAS',   required=False, type=int,   default=10,     help='Maximum number of epochs for AS' )
    parser.add_argument('--load_weights',   required=False, type=str,   default=None,     help='weights to load' )
    parser.add_argument('--model_name',   required=False, type=str,   default="Pooling",     help='name of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test", "challenge"], help='list of split for testing')


    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=3,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--model',       required=False, type=str,   default="SSModel", help='How to pool' )
    parser.add_argument('--hidden_d', required=False, type=int, default=512, help='Size of hidden dimension representation')
    parser.add_argument('--NMS_window',       required=False, type=int,   default=20, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.5, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=64,     help='Batch size' )
    parser.add_argument('--LRSS',       required=False, type=float,   default=1e-03, help='Learning Rate SS' )
    parser.add_argument('--LRAS',       required=False, type=float,   default=1e-05, help='Learning Rate AS' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-06, help='Learning Rate end' )
    parser.add_argument('--patience', required=False, type=int,   default=4,     help='Patience before reducing LR (ReduceLROnPlateau)' )

    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=4, help='number of worker to load data')

    # parser.add_argument('--logging_dir',       required=False, type=str,   default="log", help='Where to log' )
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)

    # os.makedirs(args.logging_dir, exist_ok=True)
    # log_path = os.path.join(args.logging_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    os.makedirs(os.path.join("SSmodels", args.model_name), exist_ok=True)
    log_path = os.path.join("SSmodels", args.model_name,
                            datetime.now().strftime('%Y-%m-%d_%H-%M-%S.log'))
    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    start=time.time()
    logging.info('Starting main function')
    main(args)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')
