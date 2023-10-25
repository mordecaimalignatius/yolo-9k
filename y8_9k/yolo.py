
from ultralytics.engine.model import Model
from ultralytics.models import yolo as ul_yolo
from ultralytics.utils import LOGGER, RANK, yaml_load
from ultralytics.utils.checks import check_yaml     # , check_pip_update_available
from ultralytics.cfg import TASK2DATA
from ultralytics.nn.tasks import attempt_load_one_weight

from .model import DetectionModel9000
from .train import DetectionTrainer9000
from .val import DetectionValidator9000
from .predict import DetectionPredictor9000


class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes"""
        yolo = ul_yolo.YOLO('yolov8n.yaml')
        return {
            **yolo.task_map,
            'detect9k': {
                'model': DetectionModel9000,
                'trainer': DetectionTrainer9000,
                'validator': DetectionValidator9000,
                'predictor': DetectionPredictor9000,
            },
        }

    def train(self, trainer=None, **kwargs):
        """
        hack around tedious custom data task issue

        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        # check_pip_update_available()

        overrides = yaml_load(check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task if self.task != 'detect9k' else 'detect']}  # method defaults
        args = {**overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
        if args.get('resume'):
            args['resume'] = self.ckpt_path

        self.trainer = (trainer or self.smart_load('trainer'))(overrides=args, _callbacks=self.callbacks)
        if not args.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP
        return self.metrics

    def force_task(self, task: str):
        """ override task definition to get y9000 mods also in standard models """
        self.task = task
