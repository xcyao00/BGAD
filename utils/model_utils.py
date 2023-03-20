import os, math
import numpy as np
import torch

RESULT_DIR = './results'
WEIGHT_DIR = './weights'
MODEL_DIR  = './models'

__all__ = ('save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, output_dir, exp_name, model_path, class_name):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    save_dir = os.path.join(output_dir, exp_name, 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    fp = open(os.path.join(save_dir, '{}.txt'.format(model_path)), "w")
    fp.write(result)
    fp.close()


def save_weights(encoder, decoders, output_dir, exp_name, model_path):
    model_dir = os.path.join(output_dir, exp_name, 'weights')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    state = {'encoder_state_dict': encoder.state_dict(),
             'decoder_state_dict': [decoder.state_dict() for decoder in decoders] if isinstance(decoders, list) else decoders.state_dict()}
    filename = '{}.pt'.format(model_path)
    torch.save(state, os.path.join(model_dir, filename))
    print('Saving weights to {}'.format(os.path.join(model_dir, filename)))


def load_weights(encoder, decoders, filename):
    #path = os.path.join(WEIGHT_DIR, filename)
    state = torch.load(filename)
    encoder.load_state_dict(state['encoder_state_dict'], strict=False)
    decoders = [decoder.load_state_dict(state, strict=False) for decoder, state in zip(decoders, state['decoder_state_dict'])]
    print('Loading weights from {}'.format(filename))


def adjust_learning_rate(c, optimizer, epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.scaled_lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate
