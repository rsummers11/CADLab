#-----------------------------------------------
# Preprocess Data testing code
# include/tdata
# src/tdata
#-----------------------------------------------
import numpy as np
from time import time, asctime, localtime, strftime
from util import *


class TestTransformData:
    def __init__( self ):
        self.model_name = "ConvNet"
        self.import_model()
        data_path = '../cifar-10-py-colmajor'
        data_dic = unpickle( data_path + '/' + 'data_batch_1' )
        self.data = np.require( data_dic['data'], dtype=np.single, requirements='C')
        self.label = np.require( np.array(data_dic['labels']).reshape( (1, self.data.shape[1]) ), 
                dtype=np.single, requirements='C' )

    def import_model(self):
        print "========================="
        print "Importing %s C++ module" % ('_' + self.model_name)
        self.libmodel = __import__('_' + self.model_name) 
        print "========================="

    def init_model_lib(self):
        self.libmodel.initModel(self.layers,128, 0 )

    def transform_data( self, data, scale, rotate ):
        self.libmodel.preprocess( data, 32, 32, 3, scale, rotate );

    def test(self):
        #import pdb; pdb.set_trace();
        print self.data
        index = 0
        #plot data
        plot_col_image( self.data[:,index], 32, 32, 3, "before" )

        compute_time_py = time()
        # transform data
        self.transform_data( [self.data], 0.15, 15 )
        print "(%.3f sec)" % ( time() - compute_time_py)

        # plot data again
        plot_col_image( self.data[:,index], 32, 32, 3, "after" )

        pass

def main():
    data = TestTransformData()
    data.test()

if __name__ == "__main__":
    main()
