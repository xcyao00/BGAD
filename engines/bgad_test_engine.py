import os
import math
import timm
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import t2np, get_logp, load_weights
from datasets import create_test_data_loader
from models import positionalencoding2d, load_flow_model
from utils.visualizer import plot_visualizing_results
from utils.utils import calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds


def validate(args, data_loader, encoder, decoders):
    print('\nCompute loss and scores on category: {}'.format(args.class_name))
    
    decoders = [decoder.eval() for decoder in decoders]
    
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            if args.vis:
                image_list.extend(t2np(image))
                file_names.extend(file_name)
                img_types.extend(img_type)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            
            image = image.to(args.device) # single scale
            features = encoder(image)  # BxCxHxW
            for l in range(args.feature_levels):
                e = features[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
               
                # (bs, 128, h, w)
                pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                decoder = decoders[l]

                if args.flow_arch == 'flow_model':
                    z, log_jac_det = decoder(e)  
                else:
                    z, log_jac_det = decoder(e, [pos_embed, ])

                logps = get_logp(dim, z, log_jac_det)  
                logps = logps / dim  
                logps_list[l].append(logps.reshape(bs, h, w))
    
    scores = convert_to_anomaly_scores(args, logps_list)
    # calculate detection AUROC
    img_scores = np.max(scores, axis=(1, 2))
    gt_label = np.asarray(gt_label_list, dtype=np.bool)
    #img_auc = roc_auc_score(gt_label, img_scores)
    img_auc = -1
    # calculate segmentation AUROC
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
    pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    pix_pro = -1
    if args.pro:
        pix_pro = calculate_pro_metric(scores, gt_mask)
    
    if args.vis:
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        save_dir = os.path.join('vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, 
                                 img_threshold, save_dir, file_names, img_types)

    return img_auc, pix_auc, pix_pro


def test(args):
    # Feature Extractor
    encoder = timm.create_model(args.backbone_arch, features_only=True, 
                out_indices=[i+1 for i in range(args.feature_levels)], pretrained=True)
    encoder = encoder.to(args.device).eval()
    feat_dims = encoder.feature_info.channels()
    
    # Normalizing Flows
    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]

    # data loaders
    test_loader = create_test_data_loader(args)
    
    checkpoint = os.path.join(args.checkpoint, 'weights', f'{args.dataset}_tf_efficientnet_b6_conditional_flow_model_{args.class_name}.pt')
    if args.checkpoint:
        load_weights(encoder, decoders, checkpoint)

    img_auc, pix_auc, pix_pro = validate(args, test_loader, encoder, decoders)
   
    print('{} Image AUC: {}'.format(args.class_name, img_auc * 100))
    print('{} Pixel AUC: {}'.format(args.class_name, pix_auc * 100))
    print('{} Pixel PRO: {}'.format(args.class_name, pix_pro * 100))
    
    return img_auc, pix_auc, pix_pro