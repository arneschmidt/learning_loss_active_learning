# output:  dataset_labeled, dataset_unlabeled, dataset_test, start_ids
import os

import numpy as np
import pandas as pd
import albumentations as albu

from data.data_utils import extract_wsi_df_info, extract_df_info, clean_wsi_df, get_start_label_ids
from data.dataloader import PandaDataset


class DatasetPreparator():
    def __init__(self, config):
        self.config = config
        self._load_dataframes()
        self.class_no = 4
        
    def _load_dataframes(self):
        wsi_df = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "wsi_labels.csv"))
        wsi_df = extract_wsi_df_info(wsi_df)
        self.wsi_df = wsi_df
        train_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "train_patches.csv"))
        self.train_df = extract_df_info(train_df_raw, self.wsi_df, self.config['data'], split='train')
        test_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "test_patches.csv"))
        val_df_raw = pd.read_csv(os.path.join(self.config['data']["data_split_dir"], "val_patches.csv"))
        self.val_df = extract_df_info(val_df_raw, self.wsi_df, self.config['data'], split='val')
        self.test_df = extract_df_info(test_df_raw, self.wsi_df, self.config['data'], split='test')
        self.wsi_df = clean_wsi_df(self.wsi_df, self.train_df, self.val_df, self.test_df)
        self.start_ids, _ = get_start_label_ids(self.config['random_seed'], self.train_df, self.wsi_df, self.config['data'])

    def get_datasets(self):
        panda_train = PandaDataset(self.config['data']['dir'], self.train_df['image_path'], self.train_df['class'],
                                   transform=self.get_train_transform(), target_transform=self.target_transform)
        panda_unlabeled = PandaDataset(self.config['data']['dir'], self.train_df['image_path'], self.train_df['class'],
                                       transform=self.get_test_transform(), target_transform=self.target_transform)
        panda_test = PandaDataset(self.config['data']['dir'], self.test_df['image_path'], self.test_df['class'],
                                  transform=self.get_test_transform(), target_transform=self.target_transform)

        return panda_train, panda_unlabeled, panda_test, self.start_ids

    def get_train_transform(self):
        train_transform = [
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.Blur(blur_limit=7, p=0.8),
            albu.RandomBrightnessContrast(brightness_limit=0.2,
                                          contrast_limit=0.2,
                                          p=1.0),
            albu.HueSaturationValue(hue_shift_limit=20,
                                    sat_shift_limit=20,
                                    p=1.0),
        ]
        composed_transform = albu.Compose(train_transform)

        return composed_transform

    def get_test_transform(self):
        return None

    def target_transform(self, label):
        # label = [(label == v) for v in list(range(self.class_no))]
        # label = np.stack(label, axis=-1).astype('float')
        return label
