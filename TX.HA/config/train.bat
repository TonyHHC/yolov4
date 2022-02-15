echo off
set "startTime=%time: =0%"
echo on
D:/Project/Tony/Python/yolov4/darknet/darknet/build/darknet/x64\darknet.exe detector train D:/Project/Tony/Python/yolov4/TX.HA/config/tx.data D:/Project/Tony/Python/yolov4/TX.HA/config/tx.cfg -dont_show -mjpeg_port 8090 -clear -gpus 0
echo off
set "endTime=%time: =0%"
echo "*****************************"
echo Start:    %startTime%
echo End:      %endTime%
echo "*****************************"
echo on
