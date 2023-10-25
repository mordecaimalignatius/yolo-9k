# Ultralytics YOLO 🚀, AGPL-3.0 license

import os

from ultralytics.data import converter
from ultralytics.utils import LOGGER
from ultralytics.models.yolo.detect.val import DetectionValidator

from .ops import non_max_suppression
from .utils import get_class_tree
from .metrics import ConfusionMatrix9000


class DetectionValidator9000(DetectionValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader=dataloader, save_dir=save_dir, pbar=pbar, args=args, _callbacks=_callbacks)
        # this is hacky. is this necessary might be issue with yolov5 final validation?
        self.args.task = 'detect9k'
        self.tree = None
        self.confusion_title = None

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')  # is COCO
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix9000(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []

        # get tree information from data
        self.tree = None
        tree = getattr(model, 'tree', None)
        if tree is not None:
            # model from training run
            self.tree = tree
        elif 'tree' in self.data:
            # model from validation
            self.tree = get_class_tree(list(self.data['names'].values()), self.data['tree'], self.device)

        # model abuse to pass confusion matrix plot title to validator
        self.confusion_title = getattr(model.model, 'confusion_title', None)

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return non_max_suppression(preds,
                                   self.args.conf,
                                   self.args.iou,
                                   labels=self.lb,
                                   multi_label=True,
                                   agnostic=self.args.single_cls,
                                   max_det=self.args.max_det,
                                   tree=self.tree)

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot,
                                           title=self.confusion_title)

        # print also in final eval in training (or self.args.plots)
        if not self.training or self.args.plots:
            self.confusion_matrix.metrics(names=self.names.values())
