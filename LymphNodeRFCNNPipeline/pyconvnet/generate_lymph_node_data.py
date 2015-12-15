import os
import cPickle
import numpy as np
import Image

#from https://code.google.com/p/cuda-convnet/wiki/Data
#for training a network it may not be necessary to rotate and flip the data, but I found it was necessary if I wanted to visualize the data (after it was already in the correct format) using this function:
def showImage( dictionary, outpath, imSize ):
    images = dictionary.get('data') 
    singleImage = images[:,1]

    recon = np.zeros( (imSize, imSize, 3), dtype = np.uint8 ) 
    singleImage = singleImage.reshape( (imSize*3, imSize))

    red = singleImage[0:imSize,:] 
    blue = singleImage[imSize:2*imSize,:] 
    green = singleImage[2*imSize:3*imSize,:]

    recon[:,:,0] = red 
    recon[:,:,1] = blue 
    recon[:,:,2] = green

    img = Image.fromarray(recon) 
    img.save(outpath)

def makeBatch (load_path, save_path, class_list):
    data = []
    filenames = []
    file_list = os.listdir(load_path)
    for item in  file_list:
        if item.endswith(".jpg"):
            n = os.path.join(load_path, item)
            input = Image.open(n)
            arr = np.array(input, order='C')
            im = np.fliplr(np.rot90(arr, k=3))
            data.append(im.T.flatten('C'))
            filenames.append(item)
    data = np.array(data)
    out_file = open(save_path, 'w+')
    flipDat = np.flipud(data)
    rotDat = np.rot90(flipDat, k=3)
    dic = {'batch_label':'batch 1 of 3', 'data':rotDat, 'labels':class_list, 'filenames':filenames}
    cPickle.dump(dic, out_file)
    out_file.close()
	
#run batch generator
load_path="D:/HolgerRoth/DropConnect/MyDropConnect/data/test/image_files"
class_list="D:/HolgerRoth/DropConnect/MyDropConnect/data/test/classes.csv" 

save_path="D:/HolgerRoth/DropConnect/MyDropConnect/data/test/data_batch_1" 

print load_path
print save_path
print class_list
makeBatch(load_path, save_path, class_list)
print "finished."
	