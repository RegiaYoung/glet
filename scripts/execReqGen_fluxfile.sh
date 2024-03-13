#!/bin/bash

if [ -z "$1" ]
then
    echo "please specify task to execute"
    exit 1
fi
task=$1

if [ -z "$2" ]
then
    echo "please specify # of reqeusts to provide"
    exit 1
fi
req=$2

if [ -z "$3" ]
then
    echo "please specify batch size"
    exit 1
fi
batch=$3

if [ -z "$4" ]
then
    echo "please specify rate (in seconds)"
    exit 1
fi
rate=$4

if [ -z "$5" ]
then
    echo "exp flag not specified, using UNIFORM distribution"
    dist="uni"
else
    echo "exp flag specified as "$5 ""
    if [ "$5" == 0 ];
    then
	    dist="uni"
	    echo "using UNIFORM"
    else 
	    dist="exp"
	    echo "using EXPONENTIAL"
    fi
fi
if [ -z "$6" ]
then
    echo "flux flag not specified"
    flux_flag=0
    flux_file='no_file'

else
    echo "FLUX FLAG specified!!"
    flux_flag=1
    flux_file=$6
fi

RES_DIR=$PWD/../resource
CONFIG_FILE='config.json'
DATA_ROOT_DIR=$RES_DIR

BUILD_DIR=$PWD/../bin

ADDR=10.0.0.12
#ADDR=143.248.188.238
#ADDR=192.168.0.98


if [ "$task" == "ssd-mobilenetv1" -o "$task" == "traffic" ];
then
input_txt='input-camera.txt'

else
input_txt='input.txt'

fi

$BUILD_DIR/client --task $task --hostname $ADDR  --portno 8080 \
        --requests $req --batch $batch --rate $rate \
        --input $RES_DIR/$input_txt --skip_resize 1 --root_data_dir $DATA_ROOT_DIR \
        --dist $dist --flux $flux_flag --flux_file $flux_file
