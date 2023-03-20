import numpy as np
from torch.utils.data import Sampler


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


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.randomGenerator(self.dataset.normal_idx)
        self.outlier_generator = self.randomGenerator(self.dataset.anomaly_idx)
        # n_normal: 2/3; n_outlier: 1/3
        if self.cfg.num_anomalies != 0:
            self.n_normal = 2 * self.cfg.batch_size // 3
            self.n_anomaly = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_anomaly = 0

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i
    
    def __len__(self):
        return self.cfg.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_anomaly):
                batch.append(next(self.outlier_generator))

            yield batch