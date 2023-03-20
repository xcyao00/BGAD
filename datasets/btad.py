import os
import cv2
import random
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from .utils import excluding_images


BTAD_CLASS_NAMES = ['01', '02', '03']


class BTADDataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, BTAD_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        # load dataset
        if excluded_images is not None:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
            self.x, self.y, self.mask, self.img_types = excluding_images(self.x, self.y, self.mask, self.img_types, excluded_images)
        else:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]

        x = Image.open(img_path).convert('RGB')
        
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path[:-4]), img_type

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['ok'] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                 for img_fname in img_fname_list]
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(types)


class BTADFSDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_only = False
        self.anomaly_nums = c.num_anomalies
        self.normal_nums = 'all'
        self.reuse_times = 5
        
        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.reuse_times
        self.a_labels = self.a_labels * self.reuse_times
        self.a_masks = self.a_masks * self.reuse_times
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):  # anomaly samples
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
        
        x = Image.open(img).convert('RGB')
        
        x = self.normalize(self.transform_x(x))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, label, mask

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types
        normal_count = 0
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])

            if type_ == 'ok':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:self.anomaly_nums])
                a_labels.extend([1] * self.anomaly_nums)

                gt_type_dir = os.path.join(gt_dir, type_)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:self.anomaly_nums]]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                 for img_fname in img_fname_list]
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

        # append normal images in train set
        if self.normal_nums == 'all' or normal_count < self.normal_nums:
            img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'ok')
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])
            n_img_paths.extend(img_fpath_list)
            n_labels.extend([0] * len(img_fpath_list))
            n_mask_paths.extend([None] * len(img_fpath_list))
            normal_count += len(img_fpath_list)
        if self.normal_nums != 'all':
            random.shuffle(n_img_paths)
            n_img_paths = n_img_paths[:self.normal_nums]
            n_labels = n_labels[:self.normal_nums]
            n_mask_paths = n_mask_paths[:self.normal_nums]
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class BTADFSCopyPasteDataset(Dataset):
    """
    Mvtec train dataset with anomaly samples.
    Anomaly samples: real anomaly samples,
        anomaly samples generated by copy-pasting abnormal regions to normal samples
    """
    def __init__(self, c, is_train=True):
        assert c.class_name in BTAD_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies  # number of anomaly samples in each abnormal type
        self.repeat_num = 10  # repeat times for anomaly samples
        self.reuse_times = 5  # real anomaly reuse times
        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.repeat_num
        self.a_labels = self.a_labels * self.repeat_num
        self.a_masks = self.a_masks * self.repeat_num
        # set transforms
        if is_train:
            self.transform_img = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_img = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])
        
        self.augmentors = [A.RandomRotate90(),
                A.Flip(),
                A.Transpose(),
                A.OpticalDistortion(p=1.0, distort_limit=1.0),
                A.OneOf([
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ], p=0.2),
                A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    A.IAAPiecewiseAffine(p=0.3),
                ], p=0.2),
                A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),            
                ], p=0.3),
                A.HueSaturationValue(p=0.3)]

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
    
    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):  # anomaly samples
            idx_ = idx - len(self.n_imgs)
            img, label, mask = self.a_imgs[idx_], self.a_labels[idx_], self.a_masks[idx_]
            if idx >= len(self.n_imgs) + self.anomaly_nums * self.reuse_times:
                # generating anomaly sample by copy-pasting
                img, mask = self.copy_paste(img, mask)
                img, mask = Image.fromarray(img), Image.fromarray(mask)
                img = self.normalize(self.transform_img(img))
                mask = self.transform_mask(mask)

                return img, label, mask
        else:  # normal samples
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
        img = Image.open(img).convert('RGB')
        
        img = self.normalize(self.transform_img(img))
        
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return img, label, mask
    
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmentors)), 3, replace=False)
        aug = A.Compose([self.augmentors[aug_ind[0]],
                         self.augmentors[aug_ind[1]],
                         self.augmentors[aug_ind[2]]])
        return aug

    def copy_paste(self, img, mask):
        n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
        aug = self.randAugmenter()

        image = cv2.imread(img)  # anomaly sample
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        n_image = cv2.imread(self.n_imgs[n_idx])  # normal sample
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        
        mask = Image.open(mask)
        mask = np.asarray(mask)  # (900, 900)
        
        if self.class_name == '03':
            image = image[:, 100:700, :]
            n_image = n_image[:, 100:700, :]
            mask = mask[:, 100:700]

        # augmente the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        
        # copy the augmentated anomaly area to the normal image
        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

        return n_image, aug_mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.bmp') or f.endswith('.png')])

            if type_ == 'ok':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:self.anomaly_nums])
                a_labels.extend([1] * self.anomaly_nums)

                gt_type_dir = os.path.join(gt_dir, type_)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:self.anomaly_nums]]
                if self.class_name == '03':
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.bmp')
                                 for img_fname in img_fname_list]
                else:
                    gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                    for img_fname in img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

        # append normal images in train set
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'ok')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.bmp') or f.endswith('.png')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths






