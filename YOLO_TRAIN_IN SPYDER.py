import tensorflow as tf
import cv2
from darkflow.net.build import TFNet
config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    options={'model':'G:/darkflow/cfg/yolo1.cfg',
             'load':'G:/darkflow/bin/yolo.weights',
             'batch':4,
             'epoch':200,
             'gpu':0.5,
             'train':True,
             'annotation':'G:/soccer_ball_data/annotations',
             'dataset':'G:/soccer_ball_data/images'
             }
    tfn=TFNet(options)
tfn.train()





config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    options={'model':'G:/darkflow/cfg/yolo1.cfg',
             'load':9400,
             'gpu':1.0,
             'threshold':0.01
             }
    tfn=TFNet(options)
video=cv2.VideoCapture('G:/soccer_ball_data/sample_video/test_video.mp4')
while(1):
    _,frame=video.read()
    resultinglabels=tfn.return_predict(frame)
    #print(resultinglabels)
    for z in resultinglabels:
        tlx=(z['topleft']['x'],z['topleft']['y'])
        bly=(z['bottomright']['x'],z['bottomright']['y'])
        labels=z['label']
        cv2.rectangle(frame,tlx,bly,(0,255,0),3)
        cv2.putText(frame,labels,tlx,cv2.FONT_HERSHEY_DUPLEX,1.2,(0,0,255),3)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20)&0xff==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()