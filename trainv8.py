"""
train (custom) yolo v8 model

Ludwig Mohr
ludwig.mohr@icg.tugraz.at

"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
from logging import FileHandler, Formatter

from ultralytics.utils import LOGGER

from y8_9k import YOLO


########################################################################################################################
def logger_set_handler(args: Namespace):
    """ set file handler to write logs to file """
    path = args.project
    path.mkdir(parents=True, exist_ok=True)
    path = (path / args.name).with_suffix('.txt')
    fh = FileHandler(path)
    fh.setLevel('INFO')
    fh.setFormatter(Formatter(fmt='%(message)s'))

    LOGGER.addHandler(fh)
    LOGGER.info(f'saving logs to {str(path)}')


########################################################################################################################
def main(opts: Namespace):
    # register file handler for logger to save logging output
    logger_set_handler(opts)

    model_path = Path(opts.model)
    if opts.task == 'detect9k' and not opts.resume:
        official = not (model_path.is_file() and model_path.exists())
        model = YOLO(
            model_path.with_suffix('.yaml').name if official else model_path,
            task=opts.task)
        if not opts.from_scratch and official:
            LOGGER.info('loading official pretrained weights')
            model.load(model_path.with_suffix('.pt').name)  # load pretrained weights
        elif opts.from_scratch and not official:
            # reset weights of local model (training from scratch)
            LOGGER.info('resetting model weights (training from scratch)')
            model.reset_weights()

        if opts.force:
            model.force_task(opts.task)

    else:
        model = YOLO(model_path, task=opts.task)
        if opts.from_scratch:
            LOGGER.info('resetting model weights (training from scratch)')
            model.reset_weights()

    # train the model
    if not opts.val:
        model.train(data=opts.dataset, epochs=opts.epoch, batch=opts.batch, resume=opts.resume, project=opts.project,
                    name=opts.name, optimizer=opts.optimizer, save_period=opts.save_period, patience=opts.patience,
                    imgsz=opts.imgsz,
                    )

    # final validation
    if opts.title is not None:
        # for this to work with non detect9k models the task has to be set to detect9k, providing the trained weights
        model.model.confusion_title = opts.title
    conf = {
        'split': opts.val_split,
        'conf': opts.conf,
        'iou': opts.iou,
        'project': opts.project,
        'name': opts.name,
    } if opts.val else {
        'project': model.trainer.save_dir,
        'name': 'val',
    }
    metrics = model.val(**conf)
    # print(metrics)
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

    # TODO
    # path = model.export(format="pt")


########################################################################################################################
def parse_args() -> Namespace:
    parser = ArgumentParser(description='Train YOLOv8 with class hierarchy modification')
    parser.add_argument('project', metavar='path/for/training/results', type=Path, help='output directory')
    parser.add_argument('dataset', metavar='path/to/dataset/root', type=Path, help='dataset')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=200, help='number of epochs in model training')
    parser.add_argument('-f', '--force', action='store_true', help='override task definition')
    parser.add_argument('-i', '--imgsz', type=int, default=640, help='imagesize to train/test on')
    parser.add_argument('-m', '--model', type=str, default='yolov8x', help='pre-trained weights or model definition')
    parser.add_argument('-n', '--name', type=str, default=None, help='name of experiment')
    parser.add_argument('-o', '--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
                        help='choice of optimizer')
    parser.add_argument('-p', '--patience', type=int, default=20, help='patience value for early stopping')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from last checkpoint')
    parser.add_argument('-s', '--save-period', default=-1, type=int, help='save interval of intermediate results')
    parser.add_argument('-t', '--task', type=str, default='detect9k', choices=['detect', 'detect9k'],
                        help='override task when loading from pretrained weights')

    # validation call arguments
    parser.add_argument('--conf', type=float, default=0.4, help='confidence threshold in validation')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold in validation')
    parser.add_argument('--title', type=str, default=None, help='custom experiment name for confusion matrix')
    parser.add_argument('-v', '--val', action='store_true', help='validate only, no training')
    parser.add_argument('-vs', '--val-split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='split to run validation on')

    parser.add_argument('--from-scratch', action='store_true', help='train from scratch instead of pre-trained weights')

    return parser.parse_args()


########################################################################################################################
if __name__ == '__main__':
    args = parse_args()
    main(args)
