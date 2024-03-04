from typing import Union, Tuple, List

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json

from RS2.utilities.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import \
    LimitedLenWrapper
from RS2.utilities.data_augmentation.custom_transforms.masking import MaskTransform
from RS2.utilities.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, \
    Convert3DTo2DTransform
from RS2.utilities.dataloading.data_loader_3d import nnUNetDataLoader3D
from RS2.utilities.dataloading.nnunet_dataset import nnUNetDataset
from RS2.utilities.label_handling.label_handling import LabelManager


class nnDataloader():

    def __init__(self, patch_size=None, batch_size=2, fold=4):
        if patch_size is None:
            patch_size = [128, 128, 160]
        self.folder = '/icislab/volume1/lyk/Datasets/nnUNet/nnUNet_preprocessed/Dataset002_RatsBarin/nnUNetPlans_3d_fullres'
        self.fold = fold
        self.unpack_dataset = True
        self.allowed_num_processes = 12
        self.batch_dice = False

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.initial_patch_size = (179, 217, 245)

        self.rotation_for_DA = {'x': (-0.5235987755982988, 0.5235987755982988),
                                'y': (-0.5235987755982988, 0.5235987755982988),
                                'z': (-0.5235987755982988, 0.5235987755982988)}
        self.do_dummy_2d_data_aug = False
        self.mirror_axes = (0, 1, 2)
        self.oversample_foreground_percent = 0.33

        self.label_manager = LabelManager({'background': 0, 'brain': 1}, None, False, )

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        splits_file = "/icislab/volume1/lyk/Datasets/nnUNet/nnUNet_preprocessed/Dataset002_RatsBarin/splits_final.json"

        dataset = nnUNetDataset(self.folder, case_identifiers=None,
                                num_images_properties_loading_threshold=0)
        splits = load_json(splits_file)
        # self.print_to_log_file("The split file contains %d splits." % len(splits))

        # self.print_to_log_file("Desired fold for training: %d" % fold)
        if self.fold < len(splits):
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']
            # self.print_to_log_file("This split has %d training and %d validation cases."
            #                        % (len(tr_keys), len(val_keys)))
        else:
            # self.print_to_log_file("INFO: You requested fold %d for training but splits "
            #                        "contain only %d folds. I am now creating a "
            #                        "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
            # if we request a fold that is not in the split file, create a random 80:20 split
            rnd = np.random.RandomState(seed=12345 + self.fold)
            keys = np.sort(list(dataset.keys()))
            idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
            idx_val = [i for i in range(len(keys)) if i not in idx_tr]
            tr_keys = [keys[i] for i in idx_tr]
            val_keys = [keys[i] for i in idx_val]
            # self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
            #                        % (len(tr_keys), len(val_keys)))
        # if any([i in val_keys for i in tr_keys]):
        #     self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
        #                            'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.folder, tr_keys,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.folder, val_keys,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...],
                              batch_size: int,
                              patch_size: List[int],
                              label_manager: LabelManager,
                              oversample_foreground_percent: float):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader3D(dataset_tr, batch_size,
                                   initial_patch_size,
                                   patch_size,
                                   label_manager,
                                   oversample_foreground_percent,
                                   sampling_probabilities=None, pad_sides=None)
        dl_val = nnUNetDataLoader3D(dataset_val, batch_size,
                                    patch_size,
                                    patch_size,
                                    label_manager,
                                    oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_val

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int], list[int]],
                                rotation_for_DA: dict,
                                # deep_supervision_scales: Union[List, Tuple],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None, ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        # if is_cascaded:
        #     assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
        #     tr_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))
        #     tr_transforms.append(ApplyRandomBinaryOperatorTransform(
        #         channel_idx=list(range(-len(foreground_labels), 0)),
        #         p_per_sample=0.4,
        #         key="data",
        #         strel_size=(1, 8),
        #         p_per_label=1))
        #     tr_transforms.append(
        #         RemoveRandomConnectedComponentFromOneHotEncodingTransform(
        #             channel_idx=list(range(-len(foreground_labels), 0)),
        #             key="data",
        #             p_per_sample=0.2,
        #             fill_with_other_class_p=0,
        #             dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        # if regions is not None:
        #     # the ignore label must also be converted
        #     tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
        #                                                                if ignore_label is not None else regions,
        #                                                                'target', 'target'))

        # if deep_supervision_scales is not None:
        #     tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
        #                                                       output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
            # deep_supervision_scales: Union[List, Tuple],
            foreground_labels: Union[Tuple[int, ...], List[int]] = None, ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        # if is_cascaded:
        #     val_transforms.append(MoveSegAsOneHotToData(1, foreground_labels, 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        # if regions is not None:
        #     # the ignore label must also be converted
        #     val_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
        #                                                                 if ignore_label is not None else regions,
        #                                                                 'target', 'target'))
        #
        # if deep_supervision_scales is not None:
        #     val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
        #                                                        output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_dataloaders(self):
        tr_transforms = self.get_training_transforms(self.patch_size, self.rotation_for_DA, self.mirror_axes,
                                                     self.do_dummy_2d_data_aug, order_resampling_data=3,
                                                     order_resampling_seg=1,
                                                     use_mask_for_norm=[False],
                                                     foreground_labels=[1]
                                                     )
        val_transforms = self.get_validation_transforms(foreground_labels=[1])

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=self.initial_patch_size,
                                                   batch_size=self.batch_size,
                                                   patch_size=self.patch_size,
                                                   label_manager=self.label_manager,
                                                   oversample_foreground_percent=self.oversample_foreground_percent)

        dataloader_train = LimitedLenWrapper(250, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=self.allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=True, wait_time=0.02)
        dataloader_val = LimitedLenWrapper(50, data_loader=dl_val,
                                           transform=val_transforms,
                                           num_processes=max(1, self.allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=True,
                                           wait_time=0.02)
        return [dataloader_train, dataloader_val]


if __name__ == '__main__':
    a = nnDataloader([128, 128, 128]).get_dataloaders()[0]
    i = 0
    while a.next() is not None:
        print(i)
        i = i + 1
