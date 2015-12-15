#------------------------------------------
# this script combine result of different 
#  nets and report final result
#------------------------------------------

import sys
import numpy as np
from util import unpickle

def evaluate_result( result, text ):
   # pre-condition check
   num_batches = len( result['labels'] )
   assert( num_batches == len(result['labels']) )
   # compute error
   num_cases = 0
   num_wrong = 0
   for ii in range( num_batches ):
     act_index = result['labels'][ii]
     num_cases_ii = act_index.shape[0] 
     assert( num_cases_ii == result['preds'][ii].shape[0] )
     num_cases += num_cases_ii
     pred_index = np.argmax( result['preds'][ii], 1 )
     for jj in range( num_cases_ii ):
        if pred_index[jj] != act_index[jj]:
           num_wrong += 1
   
   print text + "----Testing Error: %2.4f" % ( 1.0 *num_wrong / num_cases )
   return ( 1.0 *num_wrong / num_cases )

def main():
   num_args = len(sys.argv)
   # load result from file
   num_nets = num_args - 1

   assert( num_nets > 0 )
   errors = []

   # 0th net
   # result['labels']
   # result['preds']
   result = unpickle( sys.argv[1] ) 
   errors.append( evaluate_result( result, sys.argv[1] ) )
   num_batches = len( result['labels'] )

   #import pdb; pdb.set_trace()
   # collet all results
   for ii in range( num_nets - 1 ):
      result_ii = unpickle( sys.argv[ii+2] )
      # evaluate result_ii
      errors.append( evaluate_result( result_ii, sys.argv[ii+2] ) )
      # check num of batches is consistant
      num_batches_ii = len( result_ii['labels'] )
      for jj in range( num_batches ):
         # check label is consistant
         assert( np.array_equal( 
            result_ii['labels'][jj], result['labels'][jj] ) )
         # nc result['pred'][jj]
         result['preds'][jj] += result_ii['preds'][jj]

   # classifier mean/std accuracy
   errors = np.array( errors )
   #import pdb; pdb.set_trace()
   print "mean: " , str(100*np.mean( errors )) , " std: " , str(100*(np.std( errors )))
   # evaluate result
   evaluate_result( result, "After combine" )

if __name__ == "__main__":
   main()
