# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
from ultralytics.engine.results import Results
from ultralytics.utils import ops, DEFAULT_CFG
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from .ops import non_max_suppression


class DetectionPredictor9000(DetectionPredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super(DetectionPredictor9000, self).__init__(cfg=cfg, overrides=overrides, _callbacks=None)
        self.tree = None

    def set_tree(self, tree: dict[str, torch.Tensor]):
        self.tree = tree

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds,
                                    self.args.conf,
                                    self.args.iou,
                                    agnostic=self.args.agnostic_nms,
                                    max_det=self.args.max_det,
                                    classes=self.args.classes,
                                    tree=self.tree)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
