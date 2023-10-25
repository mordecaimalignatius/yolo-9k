"""
hierarchical class training loss implementation for yolov8

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

from ultralytics.utils.loss import v8DetectionLoss

from .tal import TaskAlignedAssigner9000


class v8DetectionLoss9000(v8DetectionLoss):
    def __init__(self, model):
        super(v8DetectionLoss9000, self).__init__(model)

        # we are lazy
        self.assigner = TaskAlignedAssigner9000(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0,
                                                tree=model.tree['tree'] if model.tree is not None else None)
