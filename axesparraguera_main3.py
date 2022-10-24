# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:11:13 2022

@author: artur
"""

import torch
import numpy as np
import logging
from dataset import SoccerNetClips, OnlineSoccerNetClips, SoccerNetClipsTesting, OnlineSoccerNetFrames, SoccerNetFramesTesting, SoccerNetFramesAudioTesting, OnlineSoccerNetFramesAudio
from model import Model, Model2, ModelFramesAudio
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from datetime import datetime
import time
from loss import CLIP_loss, NLLLoss, MaskLoss, NLLLoss_weights
from train import trainerSS, trainerAS, test, testSpotting, testSpotting2, testSpotting3

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
            dataset_Train = OnlineSoccerNetFramesAudio(path_frames = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames', 
                        path_audio = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',  
                        path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                        path_store = '/data-local/data1-ssd/axesparraguera/SoccerNetSamples',
                        features_audio = 'audio', 
                        split=["train"], framerate=args.framerate, chunk_size=args.chunk_size, framestride = args.framestride, store = False)

            dataset_Valid = OnlineSoccerNetFramesAudio(path_frames = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames', 
                        path_audio = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',  
                        path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                        path_store = '/data-local/data1-ssd/axesparraguera/SoccerNetSamples',
                        features_audio = 'audio', 
                        split=["valid"], framerate=args.framerate, chunk_size=args.chunk_size, framestride = args.framestride, store = False)
            
            
            dataset_Valid_metric  = OnlineSoccerNetFramesAudio(path_frames = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames', 
                        path_audio = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',  
                        path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                        path_store = '/data-local/data1-ssd/axesparraguera/SoccerNetSamples',
                        features_audio = 'audio', 
                        split=["valid"], framerate=args.framerate, chunk_size=args.chunk_size, framestride = args.framestride, store = False)
            
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
    model = ModelFramesAudio(weights=args.load_weights, d=args.hidden_d, 
        chunk_size=args.chunk_size, framerate=args.framerate, p_mask=args.p_mask, 
        model=args.model, backbone = 'mobilenet', masking = 'token', framestride = args.framestride,
        K=64).cuda()
    
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

        #Special layers (smaller LR)
        parameters = []
        predictors = ['convMV1', 'convMV2', 'convMA1', 'convMA2']
        types = ['.weight', '.bias']
        layers = [pred + typ for pred in predictors for typ in types]

        for idx, (name, params) in enumerate(model.named_parameters()):
            if name in layers:
                parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr': args.LRSS * 10}]
            elif 'mobilenet' in name:
                parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr': args.LRSS / 10}]
            else:
                parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr': args.LRSS}]

        if not args.SS_not:

            if args.SS_from_last:
                checkpoint = torch.load(os.path.join('SSmodels', args.model_name, 'model.pth.tar'))
                model.load_state_dict(checkpoint['state_dict'])

            optimizer = torch.optim.Adam(parameters, lr=args.LRSS, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=1e-5, amsgrad=True)
            #optimizer = torch.optim.SGD(model.parameters(), lr=args.LR,
            #                            momentum=0.9)
        
            trainerSS(train_loader, 
                model, optimizer, criterionVA, criterionMask,
                model_name=args.model_name,
                max_epochs=args.max_epochsSS,
                momentum=.99, n_batches=16)

        
        if not args.AS_not:

            if args.SS_model == 0:
                checkpoint = torch.load(os.path.join('SSmodels', args.model_name, 'model.pth.tar'))
            else:
                checkpoint =torch.load(os.path.join('SSmodels', args.model_name, 'model_' + str(args.SS_model) + '.pth.tar'))

            model.load_state_dict(checkpoint['state_dict'])
            criterion = NLLLoss_weights()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.LRAS, 
                                betas=(0.9, 0.999), eps=args.LRe, 
                                weight_decay=1e-5, amsgrad=True)
    
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)
    
            trainerAS(train_loader,
                val_loader,
                val_metric_loader,
                model, optimizer, scheduler, criterion,
                patience=args.patience,
                model_name=args.model_name,
                max_epochs=args.max_epochsAS,
                SS_base=args.SS_model,
                n_batches=16)


    
    # For the best model only
    if args.SS_model == 0:
        checkpoint = torch.load(os.path.join("ASmodels", args.model_name, "model.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join('ASmodels', args.model_name, 'model_' + str(args.SS_model) + '.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    
    
    #print(asdf)

    # test on multiple splits [test/challenge]
    
    for split in args.split_test:

        dataset_Test  = SoccerNetFramesAudioTesting(path_frames = '/data-local/data1-ssd/axesparraguera/SoccerNetFrames', 
                path_audio = '/data-local/data1-ssd/axesparraguera/SoccerNetAudio',  
                path_labels = "/data-net/datasets/SoccerNetv2/ResNET_TF2",
                features_audio = 'audio', 
                split=["test"], framerate=args.framerate, chunk_size=args.chunk_size, framestride = args.framestride,
                stride = 0.5)
        print('Test loader')
        test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
        
        
        results_l, results_t = testSpotting3(test_loader, model=model, model_name=args.model_name, NMS_window=args.NMS_window, NMS_threshold=args.NMS_threshold,
                        framestride=args.framestride, framerate=args.framerate, chunk_size=args.chunk_size, 
                        path_frames='/data-local/data1-ssd/axesparraguera/SoccerNetFrames')
        if results_l is None:
            continue

        a_mAP = results_l["a_mAP"]
        a_mAP_per_class = results_l["a_mAP_per_class"]
        a_mAP_visible = results_l["a_mAP_visible"]
        a_mAP_per_class_visible = results_l["a_mAP_per_class_visible"]
        a_mAP_unshown = results_l["a_mAP_unshown"]
        a_mAP_per_class_unshown = results_l["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training (loose metric)")
        logging.info("a_mAP visibility all: " +  str(a_mAP))
        logging.info("a_mAP visibility all per class: " +  str( a_mAP_per_class))
        logging.info("a_mAP visibility visible: " +  str( a_mAP_visible))
        logging.info("a_mAP visibility visible per class: " +  str( a_mAP_per_class_visible))
        logging.info("a_mAP visibility unshown: " +  str( a_mAP_unshown))
        logging.info("a_mAP visibility unshown per class: " +  str( a_mAP_per_class_unshown))

        a_mAP = results_t["a_mAP"]
        a_mAP_per_class = results_t["a_mAP_per_class"]
        a_mAP_visible = results_t["a_mAP_visible"]
        a_mAP_per_class_visible = results_t["a_mAP_per_class_visible"]
        a_mAP_unshown = results_t["a_mAP_unshown"]
        a_mAP_per_class_unshown = results_t["a_mAP_per_class_unshown"]

        logging.info("Best Performance at end of training (tight metric)")
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
    parser.add_argument('--p_mask', required=False, type=float, default=0.2, help='Probability of masking tokens')
    parser.add_argument('--model_name',   required=False, type=str,   default="Pooling",     help='name of the model to save' )
    parser.add_argument('--test_only',   required=False, action='store_true',  help='Perform testing only' )
    parser.add_argument('--AS_not', required=False, action='store_true', help='Train AS model')
    parser.add_argument('--SS_not', required=False, action='store_true', help='Train SS model')
    parser.add_argument('--SS_model', required=False, type=int, default=0, help='SS model to load')
    parser.add_argument('--SS_from_last', required=False, action='store_true', help='Starts training from last checkpoint')

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')


    parser.add_argument('--version', required=False, type=int,   default=2,     help='Version of the dataset' )
    parser.add_argument('--framerate', required=False, type=int,   default=2,     help='Framerate of the input features' )
    parser.add_argument('--framestride', required=False, type=int,   default=8,     help='Framestride of the input features' )
    parser.add_argument('--chunk_size', required=False, type=int,   default=5,     help='Size of the chunk (in seconds)' )
    parser.add_argument('--model',       required=False, type=str,   default="SSModel", help='How to pool' )
    parser.add_argument('--hidden_d', required=False, type=int, default=512, help='Size of hidden dimension representation')
    parser.add_argument('--NMS_window',       required=False, type=int,   default=20, help='NMS window in second' )
    parser.add_argument('--NMS_threshold',       required=False, type=float,   default=0.5, help='NMS threshold for positive results' )

    parser.add_argument('--batch_size', required=False, type=int,   default=64,     help='Batch size' )
    parser.add_argument('--LRSS',       required=False, type=float,   default=1e-04, help='Learning Rate SS' )
    parser.add_argument('--LRAS',       required=False, type=float,   default=1e-05, help='Learning Rate AS' )
    parser.add_argument('--LRe',       required=False, type=float,   default=1e-07, help='Learning Rate end' )
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
