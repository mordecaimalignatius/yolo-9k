# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from copy import copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK

from .model import DetectionModel9000
from .val import DetectionValidator9000
from .utils import get_class_tree


class DetectionTrainer9000(DetectionTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if 'tree' in self.data:
            self.model.tree = get_class_tree(list(self.model.names.values()), self.data['tree'], self.device)

        else:
            LOGGER.warning('hierarchical class trainer, but no class hierarchy in data config')

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel9000(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return DetectionValidator9000(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
