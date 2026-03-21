# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS


@VISUALIZERS.register_module()
class Det3DLocalVisualizer(Visualizer):
    """Local visualizer for 3D object detection.

    Extends the base MMEngine ``Visualizer`` so that it can be referenced
    from config files as ``type='Det3DLocalVisualizer'`` and built by
    ``Runner.from_cfg()``.

    Args:
        name (str): Name of the visualizer. Defaults to ``'visualizer'``.
        vis_backends (list[dict] or None): Visualization backend configs.
            Defaults to None.
        save_dir (str or None): Directory to save visualization results.
            Defaults to None.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 vis_backends: Optional[List[dict]] = None,
                 save_dir: Optional[str] = None,
                 **kwargs):
        super().__init__(
            name=name,
            vis_backends=vis_backends,
            save_dir=save_dir,
            **kwargs)
