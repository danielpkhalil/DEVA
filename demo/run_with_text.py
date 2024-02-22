import os
from os import path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import numpy as np

from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.ext.with_text_processor import get_mask_from_process_frame_with_text

from tqdm import tqdm
import json

from deva.model.network import DEVA

def run_demo(chunk_size, img_path, amp, temporal_setting, size, output, prompt, DINO_THRESHOLD):
    # Manually prepare the cfg dictionary
    cfg = {
        'chunk_size': chunk_size,
        'img_path': img_path,
        'amp': amp,
        'temporal_setting': temporal_setting,
        'size': size,
        'output': output,
        'prompt': prompt,
        # Add other parameters as needed
        'model': './saves/DEVA-propagation.pth',
        'save_all': False,
        'key_dim': 64,
        'value_dim': 512,
        'pix_feat_dim': 512,
        'disable_long_term': False,
        'max_mid_term_frames': 10,
        'min_mid_term_frames': 5,
        'max_long_term_elements': 10000,
        'num_prototypes': 128,
        'top_k': 30,
        'mem_every': 5,
        'GROUNDING_DINO_CONFIG_PATH': './saves/GroundingDINO_SwinT_OGC.py',
        'GROUNDING_DINO_CHECKPOINT_PATH': './saves/groundingdino_swint_ogc.pth',
        'DINO_THRESHOLD': DINO_THRESHOLD,
        'DINO_NMS_THRESHOLD': 0.8,
        'SAM_ENCODER_VERSION': 'vit_h',
        'SAM_CHECKPOINT_PATH': './saves/sam_vit_h_4b8939.pth',
        'SAM_NUM_POINTS_PER_SIDE': 64,
        'SAM_NUM_POINTS_PER_BATCH': 64,
        'SAM_PRED_IOU_THRESHOLD': 0.88,
        'SAM_OVERLAP_THRESHOLD': 0.8,
        'detection_every': 5,
        'num_voting_frames': 3,
        'max_missed_detection_count': 10,
        'max_num_objects': -1,
        'sam_variant': 'original'
    }

    # Load our checkpoint
    network = DEVA(cfg).cuda().eval()
    if cfg['model'] is not None:
        model_weights = torch.load(cfg['model'])
        network.load_weights(model_weights)
    else:
        print('No model loaded.')

    gd_model, sam_model = get_grounding_dino_model(cfg, 'cuda')
    """
    Temporal setting
    """
    cfg['temporal_setting'] = cfg['temporal_setting'].lower()
    assert cfg['temporal_setting'] in ['semionline', 'online']

    # get data
    video_reader = SimpleVideoReader(cfg['img_path'])
    loader = DataLoader(video_reader, batch_size=None, collate_fn=no_collate, num_workers=8)
    out_path = cfg['output']

    # Start eval
    vid_length = len(loader)
    # no need to count usage for LT if the video is not that long anyway
    cfg['enable_long_term_count_usage'] = (
        cfg['enable_long_term']
        and (vid_length / (cfg['max_mid_term_frames'] - cfg['min_mid_term_frames']) *
             cfg['num_prototypes']) >= cfg['max_long_term_elements'])

    print('Configuration:', cfg)

    deva = DEVAInferenceCore(network, config=cfg)
    deva.next_voting_frame = cfg['num_voting_frames'] - 1
    deva.enabled_long_id()
    result_saver = ResultSaver(out_path, None, dataset='demo', object_manager=deva.object_manager)

    masks = []

    with torch.cuda.amp.autocast(enabled=cfg['amp']):
        for ti, (frame, im_path) in enumerate(tqdm(loader)):
            mask = get_mask_from_process_frame_with_text(deva, gd_model, sam_model, im_path, result_saver, ti, image_np=frame)
            masks.append(mask)
        flush_buffer(deva, result_saver)
    result_saver.end()

    # save this as a video-level json
    with open(path.join(out_path, 'pred.json'), 'w') as f:
        json.dump(result_saver.video_json, f, indent=4)  # prettier json

    return masks

#run_demo(4, './example/vipseg/images/12_1mWNahzcsAc', True, 'semionline', 480, './example/output', 'person.hat.horse', 0.5)
