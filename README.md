# Masked-Face-Detection
This is a project of Pattern Recognition course in Tsinghua University: Object detection task of masked face based on YOLO-LITE model. The model trains and infers both on CPU. The default parameters for anchor box in the code are not so good and the mAP@.5 is just 15% for a small dataset, which means the overfitting is severe. You may change the parameters by yourself and train the model for a larger dataset to gain a better performance. 

My computer is of Intel(R) Core(TM) i7-4600 CPU @ 2.10GHz. And the inference speed of the YOLO-LITE model could be as high as 29 FPS. The entire model is trained from random Initialization, The total time used for training is about 14 hours.

@Author: Ruichen Ni, PhD, School of Areospace, Tsinghua University

@E-mail: thurcni@163.com

## Environment Requirement
1. python 3.7.x
2. torch 1.1.0
3. matplotlib
4. numpy
5. python-opencv (module cv2)
6. PIL

## Step of Inference
I have put 10 images of both masked face and unmasked face under the folder "./test_images". You just need to run the following commands:

    $ python ./inference.py

## Step of Training
### Step 1: train data download
Download the mini-train dataset from Tsinghua Cloud ( https://cloud.tsinghua.edu.cn/f/29f2223b1f6b490ab2a2/ ). Unzip the file and put the corresponding files under the folder "./codes", and gaurantee the directory is organized as follows:

    -codes
        -minitrain_resized
            -xxx.jpg
            -xxx.xml
            ...
        -minival_resized
            -xxx.jpg
            -xxx.xml
            ...
        -minitest
            -xxx.jpg
            -xxx.xml
            ...

### Step 2: find the best learning rate
$ \lambda_{coord}=5 $,$ \lambda_{no\_obj}=0.001 $; the beginning learning rate is 1e-4, and multiplies 0.8 every 5 Epoches. The model is trained for 100 Epoches.

The parameters of this step have already been set in the file "train.py", so just run the following commands:

    $ python ./train.py

### Step 3: train with a fixed learning rate
$ \lambda_{coord}=5$,$\lambda_{no\_obj}=0.001$; the learning rate is set to 1e-5, and the model is trained for 100 Epoches.

In this step you need to change the codes in "train.py" by yourself first:

1. Comment the code of line 21:

        # model = YOLONet()

2. Add the following code line (I have write this code in line 22 and comment it in the step 2, so you just need to uncomment it):

        model = torch.load("./temp/model/model_100.pkl")

3. Change the learning rate in optimizer to 1e-5:

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0005)

4. Comment the scheduler in line 24

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

5. Comment the code of line 45

        # scheduler.step()

6. Change the i+1 to i+101 in line 64

        torch.save(model, "./temp/model/model_"+str(i+101)+".pkl")

7. Run the file

        $ python ./train.py

### Step 4: train the model with a larger $\lambda_{no\_obj}$
$ \lambda_{coord}=5$,$\lambda_{no\_obj}=0.01$; the beginning learning rate is 1e-5, and multiplies 0.95 every 10 Epoches, and the model is trained for 100 Epoches.

Also, you need to modify the "train.py" first:

1. Load the model of "model_200.pkl" in line 22

        model = torch.load("./temp/model/model_200.pkl")

2. Change the i+101 to i+201 in line 64

        torch.save(model, "./temp/model/model_"+str(i+201)+".pkl")

3. Change the lambda_non_ob to 0.01 in line 78

        lambda_non_ob = torch.tensor(0.01)

4. Run the file

        $ python ./train.py

### Step 5: train the model with a much larger $\lambda_{no\_obj}$
$ \lambda_{coord}=5$,$\lambda_{no\_obj}=0.1$; the learning rate is set to 1e-5, and the model is trained for 100 Epoches.

Also, you need to modify the "train.py" first:

1. Load the model of "model_300.pkl" in line 22

        model = torch.load("./temp/model/model_300.pkl")

2. Uncomment the code of line 24, and change the corresponding parameters to:

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)  

3. Uncomment the code of line 45

        scheduler.step()

4. Change the i+201 to i+301 in line 64

        torch.save(model, "./temp/model/model_"+str(i+301)+".pkl")

5. Change the lambda_non_ob to 0.1 in line 78

        lambda_non_ob = torch.tensor(0.1)

6. Run the file

        $ python ./train.py
