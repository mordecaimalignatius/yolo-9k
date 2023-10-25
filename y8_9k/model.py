"""
custom YOLO model with yolo9000-like (but only shallow) hierarchical class training


Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

from ultralytics.nn.tasks import DetectionModel

from .loss import v8DetectionLoss9000


class DetectionModel9000(DetectionModel):
    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
        super(DetectionModel9000, self).__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.tree = None
        self.root = None
        self.confusion_title = None     # abuse of model to pass title to confusion matrix in final validation

    def init_criterion(self):
        return v8DetectionLoss9000(self)

    def set_title(self, title: str):
        self.confusion_title = title

