# YOLOv5 and YOLOv8 modified for YOLO9000-like (but shallow) hierarchical class training

Modifications of YOLO object detectors to reintroduce a shallow version of class hierarchies first presented by YOLO9000.

The implementation based on the original YOLOv5 implementation performs better than the one based on YOLOv8.

Please note that (at the time of implementation/testing) YOLOv8 did not provide an official implementation for large (i.e. 1280 x 960 pixels) images and the YOLOv8 version of the v5 models have an implementation issue in the large-image version in evaluation:

    RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 32 but got size 31 for tensor number 1 in the list.

## Dataset Prerequisites
This project works with standard YOLO-type datasets, which need to have a class hierarchy defined in the dataset configuration like this:

    tree: ['', '', <parent_0>, <parent_0>, <parent_1>, <parent_1>]

where the class tree uses the same order as the class names in the dataset configuration.
A root class is defined using an empty string, a leaf class is defined by the name of the root.
At this time, only a *shallow* tree with roots and leaves (no intermediate branches) is implemented.

If you want to test this with an existing dataset feel free to download our Crossroad Camera Dataset of Mobility Aid Users with the meta files for hierarchical class training:

[Crossroad Camera Dataset](https://repository.tugraz.at/records/2gat1-pev27)

## Training
### YOLOv5
    python trainv5.py --weights yolov5l.pt --cfg ./yolov5/models/yolov5l.yaml --batch-size 32 --imgsz 640 --optimizer AdamW --data <dataset configuration.yaml> --patience 20 --epochs 500 --seed 0 --project <output project path> --name <name>

It may happen that (depending on the random seed used) NaN values will be encountered in the loss.
If this happens, as a workaround training will be automatically restarted from the best performing checkpoint, reseeding the random seeds.
You may either accept this solution or search for a random seed for which this does not happen.

### YOLOv8
    python trainv8.py <project directory> <dataset configuration.yaml> --model yolov8l --name <project name> --batch 32 --imgsz 640

## Testing
### YOLOv5
    python valv5.py --task test --data <dataset configuration.yaml> --img-size 640 --weights <path to trained weights.pt> --project <project directory> --name <project test name> --conf-thres 0.4 --iou-thres 0.45

### YOLOv8
To test YOLOv8 models you need to set the *--val* flag and set the dataset split to test on *--val-split test*.
Different object confidence and IoU thresholds can be set using *--conf \<threshold>* and *--iou \<threshold>* options.

    python trainv8.py <project directory> <dataset configuration.yaml> --val --val-split test --model <trained model weights.pt> --name <test name> --title <title of confusion matrix>

## Acknowledgement
This work uses and adapts the following implementations
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv8](https://github.com/ultralytics/ultralytics)

and uses ideas presented in:

J. Redmon and A. Farhadi,
"YOLO9000: Better, Faster, Stronger,"
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
Honolulu, HI, USA,
2017,
pp. 6517-6525,
doi: 10.1109/CVPR.2017.690.

## Citation
If you use this code in your work or project, please reference:

    @inproceedings{mohr2023mobility
      title={{A Comprehensive Crossroad Camera Dataset of Mobility Aid Users}},
      author={{Mohr, Ludwig and Kirillova, Nadezda and Possegger, Horst and Bischof, Horst}},
      booktitle={{Proceedings of the 34th British Machine Vision Conference ({BMVC})}},
      year={2023}
    }
