#!/bin/bash

echo -e "\n\n"
echo "**************************************"
echo "***                                ***"
echo "*** Open your browser to port 6006 ***"
echo "***                                ***"
echo "**************************************"
echo -e "\n\n"

/usr/local/bin/tensorboard --logdir=./logs
