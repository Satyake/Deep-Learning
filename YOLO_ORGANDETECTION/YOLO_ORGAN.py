from darkflow.net.build' import TFNet
import tensorflow as tf 
config=tf.ConfigProto(log_device_placement=True)
with tf.Session(config=config) as sess:
    options={'model':'G:/darkflow/cfg/yolo1.cfg',
             'load':'G:/darkflow/bin/yolo.weights',
             'epoch':120,
             'gpu':0.5,
             'train':True,
             'batch':4,
             'annotation':'G:/chexray/Training_YOLO/annotations',
             'dataset':'G:/chexray/Training_YOLO/images'}
    tfn=TFNet(options)
    
tfn.train()

from darkflow.net.build import TFNet
import tensorflow as tf 
config=tf.ConfigProto(log_device_placement=True)
with tf.Session(config=config) as sess:
    options={'model':'G:/darkflow/cfg/yolo1.cfg',
             'load':10440,
             'gpu':0.5,
             'threshold':0.2
             }
    tfn=TFNet(options)
import cv2
image=cv2.imread('G:/chexray/images/test.png')
result=tfn.return_predict(image)
for r in result:
    tl=r['topleft']['x'],r['topleft']['y']
    br=r['bottomright']['x'],r['bottomright']['y']
    labels=r['label']
    cv2.rectangle(image,tl,br,(0,0,255),3)
    cv2.putText(image,labels,tl,cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),1)
cv2.imshow('organ tracked',image)
