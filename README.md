forked from [kuangliu/pytorch-ssd](https://github.com/kuangliu/pytorch-ssd)

## 1、下载代码

git clone https://github.com/melonbo/power-bank-detect-on-pytorch-with-ssd.git

## 2、运行测试指令

python eval_test_set.py --annotation_folder=./test/Annotations --image_folder=./test/JPEGImages --image_set_file=./test/image_set.txt

--annotation_folder测试集标注文件夹路径

--image_folder测试图片文件夹路径

--image_set_file包含测试集图片文件名

## 3、运行过程

toPascalVocXml.py文件会将测试集转化为为PascalVoc格式，保存在当前文件夹VOC2012目录下

eval_test_set.py调用weights路径下的训练模型对测试集进行测试

defalut下测试集保存在test目录下，运行测试指令测试结果过如下：
```
python eval_test_set.py --annotation_folder=./test/Annotations --image_folder=./test/JPEGImages --image_set_file=./test/image_set.txt
./test/Annotations
./test/JPEGImages
./test/image_set.txt
rm -rf ./VOC2012
cp ./test/image_set.txt ./VOC2012/ImageSets/Main/test.txt
/media/linc/data1/work/pycode/ssd.pytorch/ssd.py:34: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
  self.priors = Variable(self.priorbox.forward(), volatile=True)
Finished loading model!
im_detect: 1/64 1.119s
im_detect: 2/64 0.046s
im_detect: 3/64 0.080s
im_detect: 4/64 0.072s
im_detect: 5/64 0.079s
im_detect: 6/64 0.084s
im_detect: 7/64 0.093s
im_detect: 8/64 0.076s
im_detect: 9/64 0.042s
im_detect: 10/64 0.052s
im_detect: 11/64 0.050s
im_detect: 12/64 0.085s
im_detect: 13/64 0.082s
im_detect: 14/64 0.084s
im_detect: 15/64 0.081s
im_detect: 16/64 0.077s
im_detect: 17/64 0.084s
im_detect: 18/64 0.090s
im_detect: 19/64 0.081s
im_detect: 20/64 0.074s
im_detect: 21/64 0.082s
im_detect: 22/64 0.091s
im_detect: 23/64 0.080s
im_detect: 24/64 0.064s
im_detect: 25/64 0.085s
im_detect: 26/64 0.077s
im_detect: 27/64 0.076s
im_detect: 28/64 0.044s
im_detect: 29/64 0.085s
im_detect: 30/64 0.063s
im_detect: 31/64 0.083s
im_detect: 32/64 0.076s
im_detect: 33/64 0.038s
im_detect: 34/64 0.089s
im_detect: 35/64 0.073s
im_detect: 36/64 0.044s
im_detect: 37/64 0.078s
im_detect: 38/64 0.079s
im_detect: 39/64 0.077s
im_detect: 40/64 0.097s
im_detect: 41/64 0.077s
im_detect: 42/64 0.080s
im_detect: 43/64 0.076s
im_detect: 44/64 0.079s
im_detect: 45/64 0.046s
im_detect: 46/64 0.075s
im_detect: 47/64 0.083s
im_detect: 48/64 0.067s
im_detect: 49/64 0.058s
im_detect: 50/64 0.060s
im_detect: 51/64 0.091s
im_detect: 52/64 0.052s
im_detect: 53/64 0.040s
im_detect: 54/64 0.054s
im_detect: 55/64 0.095s
im_detect: 56/64 0.080s
im_detect: 57/64 0.051s
im_detect: 58/64 0.081s
im_detect: 59/64 0.087s
im_detect: 60/64 0.067s
im_detect: 61/64 0.054s
im_detect: 62/64 0.089s
im_detect: 63/64 0.090s
im_detect: 64/64 0.081s
Evaluating detections
Writing core VOC results file
eval_test_set.py:194: DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
  if dets == []:
Writing coreless VOC results file
VOC07 metric? Yes
Reading annotation for 1/64
Saving cached annotations to ./VOC2012/annotations_cache/annots.pkl
AP for core = 0.8825
AP for coreless = 0.7822
Mean AP = 0.8323
~~~~~~~~
Results:
0.882
0.782
0.832
~~~~~~~~

--------------------------------------------------------------
Results computed with the **unofficial** Python eval code.
Results should be very close to the official MATLAB eval code.
--------------------------------------------------------------
```
