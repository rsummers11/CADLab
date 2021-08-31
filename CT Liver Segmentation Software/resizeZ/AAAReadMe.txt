# Software and ReadMe written by Veit Sandfort, NIH Clinical Center

resizeZsitk -h for the options

usage: resizeZsitk.py [-h] [--filelist FILELIST] [--file FILE]
                      [--outputdir OUTPUTDIR] [--zres ZRES]

optional arguments:
  -h, --help            show this help message and exit
  --filelist FILELIST   list of nii files
  --file FILE           single nii file, if given, filelist will be ignored
  --outputdir OUTPUTDIR
                        folder to store result
  --zres ZRES           target voxel size in mm


will need sitk
conda install -c simpleitk simpleitk 

# Some additional stuff from Ron Summers

# This environment has what is needed to run this program:
conda activate MO

# Sample command line to create a batch file of cases to process, resizing to a 3mm slice thickness
cat filenamelist.txt | while read line; do echo "python3 resizeZsitk.py --file $line  --outputdir ~/drdcad/ron/CTC/ThickFromThin_3mm/CT  --zres 3"; done > make_resizer.swarm

# Sample line in make_resizer.swarm
python3 resizeZsitk.py --file /home/rsummers/drdcad/ron/CTC/thinDICOMs/nii/CT/1433_20120622_7.nii.gz  --outputdir ~/drdcad/ron/CTC/ThickFromThin_3mm/CT  --zres 3

