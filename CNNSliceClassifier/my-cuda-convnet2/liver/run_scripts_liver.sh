#!/bin/bash

function pause(){
   read -p "$*"
}

################# RUN #####################
./liver/make_general_data_liver.sh
./liver/train_liver.sh

pause 'Press [Crtl+C] to exit (avoids double execution)...'

