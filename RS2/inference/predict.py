import inspect
import multiprocessing
import os
import shutil
import time
import traceback
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, \
    save_json

import RS2
from RS2.inference.export_prediction import export_prediction_from_sigmoid
from RS2.inference.sliding_window_prediction import predict_sliding_window_return_logits, compute_gaussian
from RS2.network.RSSNet import RSSNet
from RS2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from RS2.utilities.config import default_num_processes
from RS2.utilities.file_path_utilities import should_i_save_to_file
from RS2.utilities.json_export import recursive_fix_for_json_export
from RS2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from RS2.utilities.utils import create_lists_from_splitted_dataset_folder


class PreprocessAdapter(DataLoader):
    def __init__(self, list_of_lists: List[List[str]], list_of_segs_from_prev_stage_files: Union[List[None], List[str]],
                 preprocessor: DefaultPreprocessor, output_filenames_truncated: List[str],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1):
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json = \
            preprocessor, plans_manager, configuration_manager, dataset_json

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        super().__init__(list(zip(list_of_lists, list_of_segs_from_prev_stage_files, output_filenames_truncated)),
                         1, num_threads_in_multithreaded,
                         seed_for_shuffle=1, return_incomplete=True,
                         shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_lists)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        files = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        ofile = self._data[idx][2]

        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments

        data, seg, data_properites = self.preprocessor.run_case(files, seg_prev_stage, self.plans_manager,
                                                                self.configuration_manager,
                                                                self.dataset_json)

        if np.prod(data.shape) > (2e9 / 4 * 0.85):
            # we need to temporarily save the preprocessed image due to process-process communication restrictions
            np.save(ofile + '.npy', data)
            data = ofile + '.npy'

        return {'data': data, 'data_properites': data_properites, 'ofile': ofile}


def load_what_we_need(checkpoint_name):
    # we could also load plans and dataset_json from the init arguments in the checkpoint. Not quite sure what is the
    # best method so we leave things as they are for the moment.
    dataset_json = load_json(join(RS2.__path__[0], "jsons/dataset.json"))
    plans = load_json(join(RS2.__path__[0], 'jsons/plans.json'))
    plans_manager = PlansManager(plans)

    # 2024/11/20 update
    # use cpu to load the model
    checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    parameters = checkpoint['state_dict']
    configuration_manager = plans_manager.get_configuration('3d_fullres')

    inference_allowed_mirroring_axes = (0, 1, 2)

    network = RSSNet(
        img_size=(128, 128, 160),
        in_channels=1,
        out_channels=1,
        feature_size=48,
        # use_checkpoint=True,
    )
    # network = UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=[32, 64, 128, 256], strides=[2, 2, 2])
    # network = UNETR(in_channels=1, out_channels=1, img_size=[128, 128, 160])
    # network = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=1, channels=[32, 64, 128, 256], strides=[2, 2, 2])

    # network = SwinUNETR(img_size=(128, 128, 160),
    #                 in_channels=1,
    #                 out_channels=1,
    #                 feature_size=48, )

    return parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network


def predict_from_raw_data(list_of_lists_or_source_folder: Union[str, List[List[str]]],
                          output_folder: str,
                          tile_step_size: float = 0.5,
                          use_gaussian: bool = True,
                          use_mirroring: bool = True,
                          perform_everything_on_gpu: bool = True,
                          verbose: bool = True,
                          save_probabilities: bool = False,
                          overwrite: bool = True,
                          checkpoint_name: str = '/icislab/volume1/lyk/SwinUNETR/BRATS21/log2/runs_unpooling_light/result0/model_final.pt',
                          num_processes_preprocessing: int = default_num_processes,
                          num_processes_segmentation_export: int = default_num_processes,
                          num_parts: int = 1,
                          part_id: int = 0,
                          device: torch.device = torch.device('cuda')):
    if device.type == 'cuda':
        device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!

    if device.type != 'cuda':
        perform_everything_on_gpu = False

    # let's store the input arguments so that its clear what was used to generate the prediction
    my_init_kwargs = {}
    for k in inspect.signature(predict_from_raw_data).parameters.keys():
        my_init_kwargs[k] = locals()[k]
    my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
    # safety precaution.
    recursive_fix_for_json_export(my_init_kwargs)
    maybe_mkdir_p(output_folder)
    save_json(my_init_kwargs, join(RS2.__path__[0], 'jsons/predict_from_raw_data_args.json'))

    # load all the stuff we need from the model_training_output_dir
    parameters, configuration_manager, inference_allowed_mirroring_axes, plans_manager, dataset_json, network = \
        load_what_we_need(checkpoint_name)

    # sort out input and output filenames
    if isinstance(list_of_lists_or_source_folder, str):
        list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                   dataset_json['file_ending'])
    print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
    list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
    caseids = [os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for i in list_of_lists_or_source_folder]
    print(f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
    print(f'There are {len(caseids)} cases that I would like to predict')

    output_filename_truncated = [join(output_folder, i) for i in caseids]
    # remove already predicted files form the lists
    if not overwrite:
        tmp = [isfile(i + dataset_json['file_ending']) for i in output_filename_truncated]
        not_existing_indices = [i for i, j in enumerate(tmp) if not j]

        output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
        list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
        print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
              f'That\'s {len(not_existing_indices)} cases.')
        # caseids = [caseids[i] for i in not_existing_indices]

    # placing this into a separate function doesnt make sense because it needs so many input variables...
    # preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    preprocessor = DefaultPreprocessor(verbose=verbose)

    # hijack batchgenerators, yo
    # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
    # way we don't have to reinvent the wheel here.
    num_processes = max(1, min(num_processes_preprocessing, len(list_of_lists_or_source_folder)))
    ppa = PreprocessAdapter(list_of_lists_or_source_folder, [None] * len(list_of_lists_or_source_folder), preprocessor,
                            output_filename_truncated, plans_manager, dataset_json,
                            configuration_manager, num_processes)

    mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1, None, pin_memory=device.type == 'cuda',
                                 timeout=1)

    # precompute gaussian
    inference_gaussian = torch.from_numpy(
        compute_gaussian([128, 128, 160])).half()
    if perform_everything_on_gpu:
        inference_gaussian = inference_gaussian.to(device)

    # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads

    # mta = SingleThreadedAugmenter(ppa, NumpyToTensor())
    # network = network.to(device)
    # with torch.no_grad():
    #     for preprocessed in mta:
    #         data = preprocessed['data']
    #         print(data.shape)
    #         if isinstance(data, str):
    #             delfile = data
    #             data = torch.from_numpy(np.load(data))
    #             os.remove(delfile)
    #
    #         ofile = preprocessed['ofile']
    #         print(f'\nPredicting {os.path.basename(ofile)}:')
    #         print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')
    #
    #         properties = preprocessed['data_properites']
    #
    #         prediction = None
    #         overwrite_perform_everything_on_gpu = perform_everything_on_gpu
    #         if perform_everything_on_gpu:
    #             try:
    #                 # messing with state dict names...
    #                 # 在这里载入参数
    #                 network.load_state_dict(parameters)
    #
    #                 if prediction is None:
    #                     prediction = predict_sliding_window_return_logits(
    #                         network, data, num_seg_heads,
    #                         [128, 128, 160],
    #                         mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                         tile_step_size=tile_step_size,
    #                         use_gaussian=use_gaussian,
    #                         precomputed_gaussian=inference_gaussian,
    #                         perform_everything_on_gpu=perform_everything_on_gpu,
    #                         verbose=verbose,
    #                         device=device)
    #             except RuntimeError:
    #                 print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
    #                       'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
    #                 print('Error:')
    #                 traceback.print_exc()
    #                 prediction = None
    #                 overwrite_perform_everything_on_gpu = False
    #
    #         if prediction is None:
    #             network.load_state_dict(parameters)
    #             if prediction is None:
    #                 prediction = predict_sliding_window_return_logits(
    #                     network, data, num_seg_heads,
    #                     configuration_manager.patch_size,
    #                     mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
    #                     tile_step_size=tile_step_size,
    #                     use_gaussian=use_gaussian,
    #                     precomputed_gaussian=inference_gaussian,
    #                     perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
    #                     verbose=verbose,
    #                     device=device)
    #
    #         print('Prediction done, transferring to CPU if needed')
    #         prediction = prediction.to('cpu').numpy()
    #         export_prediction_from_sigmoid(prediction, properties, configuration_manager, plans_manager,
    #                                        dataset_json, ofile, save_probabilities)
    #         print(f'done with {os.path.basename(ofile)}')

    # go go go
    # spawn allows the use of GPU in the background process in case somebody wants to do this. Not recommended. Trust me.
    # export_pool = multiprocessing.get_context('spawn').Pool(num_processes_segmentation_export)
    # export_pool = multiprocessing.Pool(num_processes_segmentation_export)

    network = network.to(device)

    # print('start compile model')
    network = torch.compile(network)
    # print('finish compile model')

    # 记录开始时间
    start_time = time.time()

    with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:

        r = []
        with torch.no_grad():
            for preprocessed in mta:
                data = preprocessed['data']
                print(data.shape)
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                # proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))
                # while not proceed:
                #     sleep(0.1)
                #     proceed = not check_workers_busy(export_pool, r, allowed_num_queued=2 * len(export_pool._pool))

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                if perform_everything_on_gpu:
                    try:
                        # messing with state dict names...
                        network.load_state_dict(parameters)

                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                [128, 128, 160],
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=perform_everything_on_gpu,
                                verbose=verbose,
                                device=device)
                    except RuntimeError:
                        print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                              'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                if prediction is None:
                    network.load_state_dict(parameters)
                    if prediction is None:
                        prediction = predict_sliding_window_return_logits(
                            network, data, num_seg_heads,
                            configuration_manager.patch_size,
                            mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                            tile_step_size=tile_step_size,
                            use_gaussian=use_gaussian,
                            precomputed_gaussian=inference_gaussian,
                            perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                            verbose=verbose,
                            device=device)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()

                if should_i_save_to_file(prediction, r, export_pool):
                    print(
                        'output is either too large for python process-process communication or all export workers are '
                        'busy. Saving temporarily to file...')
                    np.save(ofile + '.npy', prediction)
                    prediction = ofile + '.npy'

                # this needs to go into background processes
                # export_prediction(prediction, properties, configuration_name, plans, dataset_json, ofile,
                #                   save_probabilities)
                print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_sigmoid, ((prediction, properties, configuration_manager, plans_manager,
                                                          dataset_json, ofile, save_probabilities),)
                    )
                )
                print(f'done with {os.path.basename(ofile)}')

                # export_prediction_from_softmax(prediction, properties, configuration_manager, plans_manager,
                #                                dataset_json, ofile, save_probabilities)

        [i.get() for i in r]

    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time} 秒")


def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Ensure that the files are in the .nii.gz format and end with _0000.nii.gz')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-m', type=str, required=False,
                        default='RS2_pretrained_model.pt',
                        help='Path of pretrained model you want to use. Default: RS2_pretrained_model.pt')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X RS2_predict [...] instead!")

    args = parser.parse_args()

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
        perform_everything_on_gpu = False
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
        perform_everything_on_gpu = True
    else:
        device = torch.device('mps')
        perform_everything_on_gpu = False

    predict_from_raw_data(args.i,
                          args.o,
                          args.step_size,
                          use_gaussian=True,
                          use_mirroring=not args.disable_tta,
                          perform_everything_on_gpu=perform_everything_on_gpu,
                          verbose=args.verbose,
                          save_probabilities=args.save_probabilities,
                          overwrite=not args.continue_prediction,
                          checkpoint_name=args.m,
                          num_processes_preprocessing=args.npp,
                          num_processes_segmentation_export=args.nps,
                          device=device)


if __name__ == '__main__':
    predict_from_raw_data('/data/linyk/data/Rat',
                          '/data/linyk/data/rat_infer',
                          0.5,
                          use_gaussian=True,
                          use_mirroring=False,
                          perform_everything_on_gpu=True,
                          verbose=True,
                          save_probabilities=False,
                          overwrite=True,
                          num_processes_preprocessing=1,
                          num_processes_segmentation_export=1,
                          checkpoint_name='/data/linyk/code/RodentSkullStripping/RS2_pretrained_model.pt'
                          )
