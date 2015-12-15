#!/bin/bash
#./lymph-nodes/balanced_files.sh
#./lymph-nodes/convertRGBtoGray.sh
#./lymph-nodes/make_batches.sh
#./run_multiple_nets_lymph_nodes.sh
#./lymph-nodes/predict_multiview.sh

## PANCREAS ##
./pancreas/make_batches_pancreas.sh
./run_multiple_nets_lymph_nodes.sh

function pause(){
   read -p "$*"
}

pause 'Press [Crtl+C] to exit (avoids double execution)...'