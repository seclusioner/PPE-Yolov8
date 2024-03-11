# PPE-Yolov8

## Introduction
This is a side project for object detection using yolo algorithm (yolov8), simliar with another side project (car counter).

We input some videos `Inputs/...` or webcam (you can modify in `main.py`) and output have two results：`Results/ori_output.mp4` and `Results/output.mp4`, because we have two models.

### Libraries version
| Library's name | Version       |
| -------------- | ------------- |
| ultralytics    | 8.1.19        |
| torch          | 2.2.1         |
| numpy          | 1.23.0        |
| opencv-python  | 4.5.4.60      |
| filterpy       | 1.4.5         |
| scikit-image   | 0.19.3        |
| lap            | 0.4.0         |
| cvzone         | 1.6.1         |

# Custom model

## Dataset
Download the dataset on `Roboflow`, this case is about Construction safety.  [Source](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28/download/yolov8)

We edit `data.yaml` file, setting some parameters according to your file path.

If you want to create your own dataset, then you can use [open source](https://github.com/HumanSignal/labelImg), just label the objects by yourself and it will generate yolo format labeling file (.txt) for your image. For more, you can check out its README file.

## Training Your Yolo model

I take some notes for training customize yolo model, you can directly see the `training_model.ipynb` file, too (for epochs=30 and imgsz=480).

### Google colab

* Preparatory work

    If you use google colab to train the model, first you have to **upload your dataset to your drive**, and create `.ipynb` file.
    
    Then you have to go to your `Files` to check if you mount your drive to this .ipynb, to let code able to use data in your drive.
    
    Clone the yolo open source code, in this case we install ultralytics (yolov8) through pip (if you training on google colab, remember to add '!' at beginning)：
    
    ``` bash
    !pip install ultralytics
    
    ```
    
    Once you install the library from github, then you can write python code to import the library you just download：
    
    ``` python
    from ultralytics import YOLO
    
    ```

* Custom Train

    To train the yolo model, first, we can test our installation is succeed or not, enter the command as below at CLI：
    
    ``` bash
    !yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
    
    ```
    
    or
    
    ``` bash
    !yolo task=detect mode=predict model=yolov8l.pt conf=0.25 source='https://ultralytics.com/images/bus.jpg'
    
    ```

    Param.：
    - task：you can change the task like segmentation according to your application.
    - mode: predict / train base on what you're going to do
    - model: there are several models you can choose
    - conf: confidence value threhold (min. is 0.25)
    - source: image used to test, it may in your root/drive or url
    
    If you execute successfully, then you have already installed. Then we can change the mode to train (not predict), it will use pre-defined structure to train your model. The parameters in video is epochs=50 and imgsz=640, the model save as `yolo_model/ppe.pt`, correspond output is `Results/ori_output.mp4`.
    
    ``` bash
    !yolo task=detect mode=train model=yolov8l.pt data=../content/drive/MyDrive/Construction_Dataset/data.yaml epochs=50 imgsz=640
    ```
    
    But my computer can not run so many epochs, so I change the parameters to epochs=30 and imgsz=480, the model save as `yolo_model/self.pt`, correspond output is `Results/output.mp4`.

    *Rmk: When I uploading self.pt file, it shows the warning that the max. recommend size is 50MB, may cause some problems, you can run .ipynb file to produce it.*

    Param.：
    - data: To import the .yaml file, here is `../content/drive/MyDrive/Construction_Dataset/data.yaml`
    - training param.: Set by yourself if you need (e.g. epochs, for more you can go to [Official document](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings))
    
    And then you can obtain some files(loss、accuarcy、confusion matrix ... etc.), and we choose `best.pt` file as model we use.

* Coding

    So you can apply your own model in your code. Maybe your code approximately like this (assume we rename `best.pt` as `custom.pt`)：
    
    ``` python
    from ultralytics import YOLO
    import cv2
    
    model = YOLO("custom.pt")
    results = model("https://ultralytics.com/images/bus.jpg", show=True)
    
    cv2.waitkey(0)
    
    ```

### Other IDE
You can create the project to train the model, and the process is similar with google colab.


## References
- [Youtube](https://www.youtube.com/watch?v=WgPbbWmnXJ8)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
