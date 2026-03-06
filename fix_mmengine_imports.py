"""
Replace all mmengine imports with mmcv 2.x equivalents.
Run: python fix_mmengine_imports.py
"""
import os
import re

# Only process files under these directories
SCAN_DIRS = ['mmdet3d', 'tools']

# Map of (old_import_line_pattern, replacement_line)
# Order matters - more specific patterns first
REPLACEMENTS = [
    # ===== import mmengine (standalone) =====
    (r'^import mmengine\.fileio as fileio$',
     'import mmcv'),
    (r'^import mmengine$',
     'import mmcv'),

    # ===== mmengine.fileio =====
    (r'^from mmengine\.fileio import load as _load$',
     'from mmcv import load as _load'),

    # ===== mmengine.registry =====
    (r'^from mmengine\.registry import Registry, build_from_cfg$',
     'from mmcv.utils import Registry, build_from_cfg'),
    (r'^from mmengine\.registry import Registry$',
     'from mmcv.utils import Registry'),
    (r'^from mmengine\.registry import build_from_cfg$',
     'from mmcv.utils import build_from_cfg'),
    (r'^from mmengine\.registry import MODELS as MMCV_MODELS$',
     'from mmcv.cnn import MODELS as MMCV_MODELS'),
    (r'^from mmengine\.registry import MODELS as MODELS_REG$',
     'from mmcv.utils import Registry; MODELS_REG = Registry("models")'),
    (r'^from mmengine\.registry import MODELS as TRANSFORMER_LAYER$',
     'from mmcv.utils import Registry; TRANSFORMER_LAYER = Registry("transformer_layer")'),
    (r'^from mmengine\.registry import MODELS as TRANSFORMER_LAYER_SEQUENCE$',
     'from mmcv.utils import Registry; TRANSFORMER_LAYER_SEQUENCE = Registry("transformer_layer_sequence")'),
    (r'^from mmengine\.registry import MODELS as TRANSFORMER$',
     'from mmcv.utils import Registry; TRANSFORMER = Registry("transformer")'),
    (r'^from mmengine\.registry import MODELS as POSITIONAL_ENCODING$',
     'from mmcv.utils import Registry; POSITIONAL_ENCODING = Registry("positional_encoding")'),
    (r'^from mmengine\.registry import MODELS as ATTENTION$',
     'from mmcv.utils import Registry; ATTENTION = Registry("attention")'),
    (r'^from mmengine\.registry import TRANSFORMS$',
     'from mmcv.utils import Registry; TRANSFORMS = Registry("pipeline")'),
    (r'^from mmengine\.registry import RUNNERS$',
     'from mmcv.utils import Registry; RUNNERS = Registry("runner")'),
    (r'^from mmengine\.registry import HOOKS$',
     'from mmcv.utils import Registry; HOOKS = Registry("hook")'),

    # ===== mmengine.model (BaseModule, ModuleList, Sequential) =====
    (r'^from mmengine\.model import BaseModule, ModuleList, Sequential$',
     'from mmcv.runner import BaseModule, ModuleList, Sequential'),
    (r'^from mmengine\.model import BaseModule, ModuleList$',
     'from mmcv.runner import BaseModule, ModuleList'),
    (r'^from mmengine\.model import BaseModule$',
     'from mmcv.runner import BaseModule'),

    # ===== mmengine.model (weight init) =====
    (r'^from mmengine\.model\.weight_init import kaiming_init$',
     'from mmcv.cnn import kaiming_init'),
    (r'^from mmengine\.model\.weight_init import normal_init$',
     'from mmcv.cnn import normal_init'),
    (r'^from mmengine\.model import constant_init, kaiming_init$',
     'from mmcv.cnn import constant_init, kaiming_init'),
    (r'^from mmengine\.model import constant_init, trunc_normal_init$',
     'from mmcv.cnn import constant_init, trunc_normal_init'),
    (r'^from mmengine\.model import xavier_init, BaseModule$',
     'from mmcv.cnn import xavier_init\nfrom mmcv.runner import BaseModule'),
    (r'^from mmengine\.model import xavier_init$',
     'from mmcv.cnn import xavier_init'),
    (r'^from mmengine\.model import normal_init$',
     'from mmcv.cnn import normal_init'),
    (r'^from mmengine\.model import kaiming_init$',
     'from mmcv.cnn import kaiming_init'),
    (r'^from mmengine\.model import bias_init_with_prob, normal_init$',
     'from mmcv.cnn import bias_init_with_prob, normal_init'),
    (r'^from mmengine\.model import force_fp32$',
     'from mmcv.runner import force_fp32'),

    # ===== mmengine.model (DDP / DataParallel) =====
    (r'^from mmengine\.model import MMDistributedDataParallel$',
     'from mmcv.parallel import MMDistributedDataParallel'),
    (r'^from mmengine\.model import MMEngineDataParallel as MMDataParallel.*$',
     'from mmcv.parallel import MMDataParallel'),
    (r'^from mmengine\.model\.utils import fuse_conv_bn$',
     'from mmcv.cnn import fuse_conv_bn'),

    # ===== mmengine.runner =====
    (r'^from mmengine\.runner import load_checkpoint as _load_checkpoint$',
     'from mmcv.runner import load_checkpoint as _load_checkpoint'),
    (r'^from mmengine\.runner import load_checkpoint$',
     'from mmcv.runner import load_checkpoint'),
    (r'^from mmengine\.runner import Runner as EpochBasedRunner.*$',
     'from mmcv.runner import EpochBasedRunner'),
    (r'^from mmengine\.runner import Runner$',
     'from mmcv.runner import EpochBasedRunner as Runner'),
    (r'^from mmengine\.runner import set_random_seed$',
     'from mmdet.apis import set_random_seed'),

    # ===== mmengine.config =====
    (r'^from mmengine\.config import Config, DictAction, ConfigDict$',
     'from mmcv import Config, DictAction\nfrom mmcv.utils import ConfigDict'),
    (r'^from mmengine\.config import ConfigDict$',
     'from mmcv.utils import ConfigDict'),
    (r'^from mmengine\.config import Config, DictAction$',
     'from mmcv import Config, DictAction'),

    # ===== mmengine.dist =====
    (r'^from mmengine\.dist import get_dist_info, init_dist$',
     'from mmcv.runner import get_dist_info\nfrom mmcv.dist import init_dist'),
    (r'^from mmengine\.dist import get_dist_info$',
     'from mmcv.runner import get_dist_info'),

    # ===== mmengine.logging =====
    (r'^from mmengine\.logging import print_log$',
     'from mmcv.utils import print_log'),
    (r'^from mmengine\.logging import MMLogger$',
     'import logging; MMLogger = logging.getLogger'),

    # ===== mmengine.utils =====
    (r'^from mmengine\.utils import is_tuple_of$',
     'from mmcv.utils import is_tuple_of'),
    (r'^from mmengine\.utils import is_list_of$',
     'from mmcv.utils import is_list_of'),
    (r'^from mmengine\.utils import multi_apply$',
     'from mmdet.core import multi_apply'),
    (r'^from mmengine\.utils\.misc import multi_apply$',
     'from mmdet.core import multi_apply'),
    (r'^from mmengine\.utils import to_2tuple$',
     'from mmcv.utils import to_2tuple'),
    (r'^from mmengine\.utils import add_prefix$',
     'def add_prefix(inputs, prefix):\n    return {f"{prefix}.{k}": v for k, v in inputs.items()}'),
    (r'^from mmengine\.utils import deprecated_api_warning$',
     'from mmcv.utils import deprecated_api_warning'),
    (r'^from mmengine\.utils import track_iter_progress$',
     'from mmcv.utils import track_iter_progress'),
    (r'^from mmengine\.utils import ProgressBar$',
     'from mmcv.utils import ProgressBar'),
    (r'^from mmengine\.utils import get_git_hash$',
     'from mmcv.utils import get_git_hash'),
    (r'^from mmengine\.utils\.dl_utils import collect_env as collect_base_env$',
     'from mmcv.utils import collect_env as collect_base_env'),
    (r'^from mmengine\.utils import import_modules_from_strings$',
     'from mmcv.utils import import_modules_from_strings'),

    # ===== mmengine.dataset =====
    (r'^from mmengine\.dataset import Compose$',
     'from mmcv.utils import build_from_cfg\n# Compose will be imported from local pipelines'),
    (r'^from mmengine\.dataset import ConcatDataset, RepeatDataset$',
     'from torch.utils.data import ConcatDataset\nfrom mmdet.datasets.dataset_wrappers import RepeatDataset'),
    (r'^from mmengine\.dataset import pseudo_collate$',
     'from mmcv.parallel import collate as pseudo_collate'),
    (r'^from mmengine\.dataset import DefaultSampler$',
     'from torch.utils.data.distributed import DistributedSampler as DefaultSampler'),

    # ===== mmengine.structures =====
    (r'^from mmengine\.structures import DataContainer as DC$',
     'from mmcv.parallel import DataContainer as DC'),

    # ===== mmengine.optim =====
    (r'^from mmengine\.optim import build_optim_wrapper$',
     'pass  # build_optim_wrapper not needed in mmcv 2.x'),
]


def fix_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    if 'mmengine' not in content:
        return False

    lines = content.split('\n')
    changed = False
    new_lines = []

    for line in lines:
        stripped = line.strip()
        matched = False
        for pattern, replacement in REPLACEMENTS:
            if re.match(pattern, stripped):
                # Preserve indentation
                indent = line[:len(line) - len(line.lstrip())]
                rep_lines = replacement.split('\n')
                for rl in rep_lines:
                    new_lines.append(indent + rl)
                matched = True
                changed = True
                break
        if not matched:
            # Handle inline mmengine references like `mmengine.load(` -> `mmcv.load(`
            if 'mmengine.load(' in line:
                line = line.replace('mmengine.load(', 'mmcv.load(')
                changed = True
            if 'mmengine.dump(' in line:
                line = line.replace('mmengine.dump(', 'mmcv.dump(')
                changed = True
            if 'fileio.get(' in line or 'fileio.load(' in line:
                line = line.replace('fileio.get(', 'mmcv.load(')
                line = line.replace('fileio.load(', 'mmcv.load(')
                changed = True
            new_lines.append(line)

    if changed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f'  FIXED: {filepath}')
    return changed


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    total_fixed = 0

    for scan_dir in SCAN_DIRS:
        dirpath = os.path.join(root, scan_dir)
        for dirp, dirs, files in os.walk(dirpath):
            for fname in files:
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(dirp, fname)
                if fix_file(fpath):
                    total_fixed += 1

    print(f'\nDone! Fixed {total_fixed} files.')


if __name__ == '__main__':
    main()
