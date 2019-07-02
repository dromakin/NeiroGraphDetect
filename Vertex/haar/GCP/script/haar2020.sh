{\rtf1\ansi\ansicpg1251\cocoartf1671\cocoasubrtf500
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww26260\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/bash\
cd dataset\
opencv_createsamples -info good.dat -vec ./2020.vec -w 20 -h 20 -num 917\
opencv_traincascade -data ./haar2020 -vec 2020.vec -bg good.dat -numStages 25 -minhitrate 0.999 -maxFalseAlarmRate 0.5 -numPos 819 -numNeg 1468 -w 20 -h 20 -mode ALL -precalcValBufSize 25000 -precalcIdxBufSize 25000}