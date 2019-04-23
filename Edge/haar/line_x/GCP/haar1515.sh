{\rtf1\ansi\ansicpg1251\cocoartf1671
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 #!/bin/bash\
cd dataset\
opencv_createsamples -info line_start_pos_big_JPG.dat -vec ./1515.vec -w 15 -h 15 -num 900\
opencv_traincascade -data ./haarline1515 -vec 1515.vec -bg line_start_neg_big_JPG.dat -numStages 25 -minhitrate 0.995 -maxFalseAlarmRate 0.5 -maxDepth 3 -numPos 700 -numNeg 1100 -w 15 -h 15 -bt RAB -mode ALL -precalcValBufSize 12000 -precalcIdxBufSize 12000\
}