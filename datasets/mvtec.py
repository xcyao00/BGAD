import os
import cv2
import glob
import random
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
import imgaug.augmenters as iaa
import albumentations as A
from .perlin import rand_perlin_2d_np
from .nsa import patch_ex
from skimage.segmentation import mark_boundaries


# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
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
        
        x = Image.open(img_path)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        
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
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(types)


class MVTecFSDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
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
        #x = Image.open(x).convert('RGB')
        x = Image.open(img)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
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
                                     if f.endswith('.png')])

            if type_ == 'good':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:self.anomaly_nums])
                a_labels.extend([1] * self.anomaly_nums)

                gt_type_dir = os.path.join(gt_dir, type_)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:self.anomaly_nums]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

        # append normal images in train set
        if self.normal_nums == 'all' or normal_count < self.normal_nums:
            img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.png')])
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


class MVTecFSCopyPasteDataset(Dataset):
    """
    Mvtec train dataset with anomaly samples.
    Anomaly samples: real anomaly samples,
        anomaly samples generated by copy-pasting abnormal regions to normal samples
    """
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies  # number of anomaly samples in each abnormal type
        self.repeat_num = 10  # repeat times for anomaly samples
        self.reuse_times = 5  # real anomaly reuse times
        self.in_fg_region = c.in_fg_region
        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.repeat_num
        self.a_labels = self.a_labels * self.repeat_num
        self.a_masks = self.a_masks * self.repeat_num
        
        self.labels = np.array(self.n_labels + self.a_labels)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.anomaly_idx = np.argwhere(self.labels == 1).flatten()
        
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
                # img.save('aug_imgs/gen_img.jpg')
                # mask.save('aug_imgs/gen_mask.png')
                img = self.normalize(self.transform_img(img))
                mask = self.transform_mask(mask)

                return img, label, mask
        else:  # normal samples
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
        img = Image.open(img)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            img = np.expand_dims(np.asarray(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.normalize(self.transform_img(img))
        #
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
        # aug = A.Compose([self.augmentors[0], A.GridDistortion(p=1.0), self.augmentors[3], self.augmentors[1], self.augmentors[7]])
        return aug

    def copy_paste(self, img, mask):
        n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
        aug = self.randAugmenter()

        image = cv2.imread(img)  # anomaly sample
        # temp_img = Image.open(img)
        # temp_img.save("aug_imgs/ano_img.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        n_image = cv2.imread(self.n_imgs[n_idx])  # normal sample
        n_image = cv2.cvtColor(n_image, cv2.COLOR_BGR2RGB)  # (900, 900, 3)
        # temp = Image.open(self.n_imgs[n_idx])
        # temp.save('aug_imgs/nor_img.jpg')
        img_height, img_width = n_image.shape[0], n_image.shape[1]

        mask = Image.open(mask)
        mask = np.asarray(mask)  # (900, 900)
        
        # augmente the abnormal region
        augmentated = aug(image=image, mask=mask)
        aug_image, aug_mask = augmentated['image'], augmentated['mask']
        # temp_img = Image.fromarray(aug_image)
        # temp_img.save("aug_imgs/ano_aug_img.jpg")
        # crop_img = aug_image.copy()
        # crop_img[aug_mask == 0] = 0
        # crop_img = Image.fromarray(crop_img)
        # crop_img.save('aug_imgs/crop_img.jpg')
        if self.in_fg_region:
            n_img_path = self.n_imgs[n_idx]
            img_file = n_img_path.split('/')[-1]
            fg_path = os.path.join(f'fg_mask/{self.class_name}', img_file)
            fg_mask = Image.open(fg_path)
            #fg_mask.save("aug_imgs/fg_mask.jpg")
            fg_mask = np.asarray(fg_mask)
            
            intersect_mask = np.logical_and(fg_mask == 255, aug_mask == 255)
            if (np.sum(intersect_mask) > int(2 / 3 * np.sum(aug_mask == 255))):
                # when most part of aug_mask is in the fg_mask region 
                # copy the augmentated anomaly area to the normal image
                n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                return n_image, aug_mask
            else:
                contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                center_xs, center_ys = [], []
                widths, heights = [], []
                for i in range(len(contours)):
                    M = cv2.moments(contours[i])
                    if M['m00'] == 0:  # error case
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        center_x = int((x_min + x_max) / 2)
                        center_y = int((y_min + y_max) / 2)
                    else:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                    center_xs.append(center_x)
                    center_ys.append(center_y)
                    x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                    y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                    width, height = x_max - x_min, y_max - y_min
                    widths.append(width)
                    heights.append(height)
                if len(widths) == 0 or len(heights) == 0:  # no contours
                    n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                    return n_image, aug_mask
                else:
                    max_width, max_height = np.max(widths), np.max(heights)
                    center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    center_mask[int(max_height/2):img_height-int(max_height/2), int(max_width/2):img_width-int(max_width/2)] = 255
                    fg_mask = np.logical_and(fg_mask == 255, center_mask == 255)

                    x_coord = np.arange(0, img_width)
                    y_coord = np.arange(0, img_height)
                    xx, yy = np.meshgrid(x_coord, y_coord)
                    # coordinates of fg region points
                    xx_fg = xx[fg_mask]
                    yy_fg = yy[fg_mask]
                    xx_yy_fg = np.stack([xx_fg, yy_fg], axis=-1)  # (N, 2)
                    
                    if xx_yy_fg.shape[0] == 0:  # no fg
                        n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]
                        return n_image, aug_mask

                    aug_mask_shifted = np.zeros((img_height, img_width), dtype=np.uint8)
                    for i in range(len(contours)):
                        aug_mask_shifted_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        new_aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        # random generate a point in the fg region
                        idx = np.random.choice(np.arange(xx_yy_fg.shape[0]), 1)
                        rand_xy = xx_yy_fg[idx]
                        delta_x, delta_y = center_xs[i] - rand_xy[0, 0], center_ys[i] - rand_xy[0, 1]
                        
                        x_min, x_max = np.min(contours[i][:, :, 0]), np.max(contours[i][:, :, 0])
                        y_min, y_max = np.min(contours[i][:, :, 1]), np.max(contours[i][:, :, 1])
                        
                        # mask for one anomaly region
                        aug_mask_i = np.zeros((img_height, img_width), dtype=np.uint8)
                        aug_mask_i[y_min:y_max, x_min:x_max] = 255
                        aug_mask_i = np.logical_and(aug_mask == 255, aug_mask_i == 255)
                        
                        # coordinates of orginal mask points
                        xx_ano, yy_ano = xx[aug_mask_i], yy[aug_mask_i]
                        
                        # shift the original mask into fg region
                        xx_ano_shifted = xx_ano - delta_x
                        yy_ano_shifted = yy_ano - delta_y
                        outer_points_x = np.logical_or(xx_ano_shifted < 0, xx_ano_shifted >= img_width) 
                        outer_points_y = np.logical_or(yy_ano_shifted < 0, yy_ano_shifted >= img_height)
                        outer_points = np.logical_or(outer_points_x, outer_points_y)
                        
                        # keep points in image
                        xx_ano_shifted = xx_ano_shifted[~outer_points]
                        yy_ano_shifted = yy_ano_shifted[~outer_points]
                        aug_mask_shifted_i[yy_ano_shifted, xx_ano_shifted] = 255
                        
                        # original points should be changed
                        xx_ano = xx_ano[~outer_points]
                        yy_ano = yy_ano[~outer_points]
                        new_aug_mask_i[yy_ano, xx_ano] = 255
                        # copy the augmentated anomaly area to the normal image
                        n_image[aug_mask_shifted_i == 255, :] = aug_image[new_aug_mask_i == 255, :]
                        aug_mask_shifted[aug_mask_shifted_i == 255] = 255
                    return n_image, aug_mask_shifted
        else:  # no fg restriction
            # copy the augmentated anomaly area to the normal image
            n_image[aug_mask == 255, :] = aug_image[aug_mask == 255, :]

            return n_image, aug_mask

    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types

        num_ano_types = len(ano_types) - 1
        anomaly_nums_per_type = self.anomaly_nums // num_ano_types
        extra_nums = self.anomaly_nums % num_ano_types
        extra_ano_img_list, extra_ano_gt_list = [], []
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            if type_ == 'good':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:anomaly_nums_per_type])
                a_labels.extend([1] * anomaly_nums_per_type)

                extra_ano_img_list.extend(img_fpath_list[anomaly_nums_per_type:])

                gt_type_dir = os.path.join(gt_dir, type_)
                ano_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:anomaly_nums_per_type]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in ano_img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

                extra_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[anomaly_nums_per_type:]]
                extra_gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in extra_img_fname_list]
                extra_ano_gt_list.extend(extra_gt_fpath_list)
        if extra_nums > 0:
            assert len(extra_ano_img_list) == len(extra_ano_gt_list)
            inds = list(range(len(extra_ano_img_list)))
            random.shuffle(inds)
            select_ind = inds[:extra_nums]
            extra_a_img_paths = [extra_ano_img_list[ind] for ind in select_ind]
            extra_a_labels = [1] * extra_nums
            extra_a_mask_paths = [extra_ano_gt_list[ind] for ind in select_ind]
            a_img_paths.extend(extra_a_img_paths)
            a_labels.extend(extra_a_labels)
            a_mask_paths.extend(extra_a_mask_paths)

        # append normal images in train set
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.png')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class MVTecPseudoDataset(Dataset):
    def __init__(self, c, is_train=True):
        """
        Mvtec train dataset with anomaly samples.
        Anomaly samples: Pseudo anomaly samples.
        """
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        self.anomaly_nums = c.num_anomalies
        self.repeat_num = 10
        self.reuse_times = 5
        self.ano_type = 'nsa'
        
        # load dataset
        self.n_imgs, self.n_labels, self.n_masks, self.a_imgs, self.a_labels, self.a_masks = self.load_dataset_folder()
        self.a_imgs = self.a_imgs * self.repeat_num
        self.a_labels = self.a_labels * self.repeat_num
        self.a_masks = self.a_masks * self.repeat_num
        
        self.labels = np.array(self.n_labels + self.a_labels)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.anomaly_idx = np.argwhere(self.labels == 1).flatten()
        
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

        self.anomaly_source_paths = sorted(glob.glob(c.anomaly_source_path + "/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        # keep same with the MVTecCopyPasteDataset
        self.transform_img_np = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                #T.RandomRotation(5),
                T.CenterCrop(c.crop_size)])
        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])
        self.normalize_np = T.Compose([T.ToTensor(), T.Normalize(c.norm_mean, c.norm_std)])

    def __len__(self):
        return len(self.n_imgs) + len(self.a_imgs)

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]])
        #aug = iaa.Sequential([self.augmenters[4], self.augmenters[3], self.augmenters[5]])
        return aug

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        # anomaly_source_path = '/disk/yxc/datasets/dtd/images/blotchy/blotchy_0069.jpg'
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.cropsize[1], self.cropsize[0]))

        # temp_img = Image.open(anomaly_source_path)
        # temp_img.save('texture_img.jpg')
        anomaly_img_augmented = aug(image=anomaly_source_img)
        # temp_img = cv2.cvtColor(anomaly_img_augmented, cv2.COLOR_BGR2RGB)
        # temp_img = Image.fromarray(temp_img)
        # temp_img.save('texture_img_aug.jpg')
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.cropsize[0], self.cropsize[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
            perlin_thr)
        # temp_img = augmented_image.astype(np.float32)
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('aug_img.jpg', temp_img, )
        # temp_img = augmented_image.astype(np.uint8)
        # temp_img = Image.fromarray(temp_img)
        # temp_img.save('aug_img.jpg')
        
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1
        if np.sum(msk) == 0:
            has_anomaly = 0
        return augmented_image, msk, has_anomaly

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path)
        #image.save('ori_img.jpg')
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            image = np.expand_dims(np.asarray(image), axis=2)
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image.astype('uint8')).convert('RGB')
        #
        image = self.transform_img_np(image)
        image = np.asarray(image)  # (256, 256, 3)

        do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        if do_aug_orig:
            image = self.rot(image=image)

        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = augmented_image.astype(np.uint8)
        augmented_image = Image.fromarray(augmented_image)
        #augmented_image.save('aug_img.jpg')
        augmented_image = self.normalize_np(augmented_image)
        anomaly_mask = torch.from_numpy(np.transpose(anomaly_mask, (2, 0, 1)))

        return augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        if idx >= len(self.n_imgs):  # anomaly samples
            if self.ano_type == 'perlin':
                n_idx = np.random.randint(len(self.n_imgs))  # get a random normal sample
                anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
                image, mask, label = self.transform_image(self.n_imgs[n_idx], 
                                                                    self.anomaly_source_paths[anomaly_source_idx])
            if self.ano_type == 'nsa':
                n_idx1 = np.random.randint(len(self.n_imgs))  # get a random normal sample
                n_idx2 = np.random.randint(len(self.n_imgs))
                dst_img = Image.open(self.n_imgs[n_idx1])
                src_img = Image.open(self.n_imgs[n_idx2])
                if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
                    dst_img = np.expand_dims(np.asarray(dst_img), axis=2)
                    dst_img = np.concatenate([dst_img, dst_img, dst_img], axis=2)
                    src_img = np.expand_dims(np.asarray(src_img), axis=2)
                    src_img = np.concatenate([src_img, src_img, src_img], axis=2)
                else:
                    dst_img = np.array(dst_img)
                    src_img = np.array(src_img)
                
                image, mask = patch_ex(dst_img, src_img)
                mask = cv2.resize(mask, dsize=(self.cropsize[1], self.cropsize[0]), interpolation=cv2.INTER_NEAREST)
                image = Image.fromarray(image).convert('RGB')
                image = self.normalize(self.transform_img(image))
                mask = torch.from_numpy(mask).unsqueeze(0).to(torch.float32)  # (1, 256, 256), float32, [0, 1]
                label = 1
            
            return image, label, mask
        else:
            img, label, mask = self.n_imgs[idx], self.n_labels[idx], self.n_masks[idx]
        img = Image.open(img)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            img = np.expand_dims(np.asarray(img), axis=2)
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        #
        img = self.normalize(self.transform_img(img))
        #
        if label == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return img, label, mask
    
    def load_dataset_folder(self):
        n_img_paths, n_labels, n_mask_paths = [], [], []  # normal
        a_img_paths, a_labels, a_mask_paths = [], [], []  # abnormal

        img_dir = os.path.join(self.dataset_path, self.class_name, 'test')
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        ano_types = sorted(os.listdir(img_dir))  # anomaly types

        num_ano_types = len(ano_types) - 1
        anomaly_nums_per_type = self.anomaly_nums // num_ano_types
        extra_nums = self.anomaly_nums % num_ano_types
        extra_ano_img_list, extra_ano_gt_list = [], []
        for type_ in ano_types:
            # load images
            img_type_dir = os.path.join(img_dir, type_)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])

            if type_ == 'good':  # normal images
                continue
            else:  # anomaly images
                # randomly choose some anomaly images
                random.shuffle(img_fpath_list)
                a_img_paths.extend(img_fpath_list[:anomaly_nums_per_type])
                a_labels.extend([1] * anomaly_nums_per_type)

                extra_ano_img_list.extend(img_fpath_list[anomaly_nums_per_type:])

                gt_type_dir = os.path.join(gt_dir, type_)
                ano_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[:anomaly_nums_per_type]]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in ano_img_fname_list]
                a_mask_paths.extend(gt_fpath_list)

                extra_img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list[anomaly_nums_per_type:]]
                extra_gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in extra_img_fname_list]
                extra_ano_gt_list.extend(extra_gt_fpath_list)
        if extra_nums > 0:
            assert len(extra_ano_img_list) == len(extra_ano_gt_list)
            inds = list(range(len(extra_ano_img_list)))
            random.shuffle(inds)
            select_ind = inds[:extra_nums]
            extra_a_img_paths = [extra_ano_img_list[ind] for ind in select_ind]
            extra_a_labels = [1] * extra_nums
            extra_a_mask_paths = [extra_ano_gt_list[ind] for ind in select_ind]
            a_img_paths.extend(extra_a_img_paths)
            a_labels.extend(extra_a_labels)
            a_mask_paths.extend(extra_a_mask_paths)

        # append normal images in train set
        img_dir = os.path.join(self.dataset_path, self.class_name, 'train', 'good')
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.png')])
        n_img_paths.extend(img_fpath_list)
        n_labels.extend([0] * len(img_fpath_list))
        n_mask_paths.extend([None] * len(img_fpath_list))
        
        return n_img_paths, n_labels, n_mask_paths, a_img_paths, a_labels, a_mask_paths


class MVTecAnomalyDataset(Dataset):
    """
    Mvtec train dataset with anomaly samples.
    Anomaly samples: Pseudo anomaly samples, real anomaly samples, 
        anomaly samples generated by copy-pasting abnormal regions to normal samples
    """
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.copy_paste_dataset = MVTecFSCopyPasteDataset(c, is_train=is_train)
        self.pseudo_dataset = MVTecPseudoDataset(c, is_train=is_train)

    def __len__(self):
        return len(self.copy_paste_dataset) + len(self.pseudo_dataset) - len(self.copy_paste_dataset.n_imgs)

    def __getitem__(self, idx):
        if idx >= len(self.pseudo_dataset):
            return self.copy_paste_dataset.__getitem__(idx)
        else:
            return self.pseudo_dataset.__getitem__(idx)


def excluding_images(images, labels, masks, img_types, excluding_images):
    retain_images = []
    retain_labels = []
    retain_masks = []
    retain_img_types = []
    for idx, image in enumerate(images):
        if image in excluding_images:
            continue
        retain_images.append(image)
        retain_labels.append(labels[idx])
        retain_masks.append(masks[idx])
        retain_img_types.append(img_types[idx])
    
    return retain_images, retain_labels, retain_masks, retain_img_types