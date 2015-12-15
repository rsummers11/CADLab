#1. Start by training the net on batches 1-4 and "testing" on batch 5. Train until the validation error stops improving. Since I've run this net before, I know that this happens after roughly 100 epochs. So we start the net with the parameters --train-range=1-4, --test-range=5, and --epochs=100. Like so:
python convnet.py --data-path=../data/cifar-10-py-colmajor/ --save-path=../data/cifar-10-py-colmajor_RESULTS --test-range=5 --train-range=1-4 --layer-def=./example-layers/layers-conv-local-13pct.cfg --layer-params=./example-layers/layer-params-conv-local-13pct.cfg --data-provider=cifar-cropped --test-freq=13 --crop-border=4 --epochs=100

#test python convnet.py --data-path=../data/cifar-10-py-colmajor/ --save-path=../data/cifar-10-py-colmajor_RESULTS --test-range=5 --train-range=1-4 --layer-def="D:\HolgerRoth\DropConnect\dropnn-exp\dropnn-exp-release\fig2\fc400-relu-dc-1.cfg" --layer-params="D:\HolgerRoth\DropConnect\dropnn-exp\dropnn-exp-release\fig2\params-2fc-mnist.cfg" --data-provider=cifar-cropped --test-freq=13 --crop-border=4 --epochs=100


# OUTPUT after first training (or similar): Now note the training error after the above run has finished. In this case, it is roughly 0.4 nats:
100.2... logprob:  0.418657, 0.145800 (2.010 sec)
100.3... logprob:  0.405222, 0.141100 (1.999 sec)
100.4... logprob:  0.419132, 0.145400 (2.001 sec)
101.1... logprob:  0.394249, 0.137100 (2.009 sec)

#3. Now we'll resume training, but this time on all 5 batches of the CIFAR-10 training set. We will stop training when the training error on batch 5 (which used to be our validation set) reaches 0.4 nats. Because I have done this before, I know this takes roughly 40 more epochs. So we run the net like this:
python convnet.py -f ../data/cifar-10-py-colmajor_RESULTS/model_fc-13 --train-range=1-5 --epochs=140

#4. Now we reduce all learning rates (the epsW parameters) in the layer parameter file by a factor of 10, and train for another 10 epochs:
python convnet.py -f ../data/cifar-10-py-colmajor_RESULTS/model_fc-13 --epochs=150

#5. Reduce all learning rates in the layer parameter file by another factor of 10, and train for another 10 epochs:
python convnet.py -f ../data/cifar-10-py-colmajor_RESULTS/model_fc-13 --epochs=160

#######################################################################
## TEST DIFFERENT NETS
#######################################################################

set NET=D:\HolgerRoth\data\LymphNodes\Abdominal_LN\LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_RESULTS\model_fc-13pct-dc
set NET=D:\HolgerRoth\data\LymphNodes\Abdominal_LN\LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_RESULTS\model_fc-conv32-dc
set NET=D:/HolgerRoth/data/LymphNodes/Abdominal_LN/LymphNodeData_highRes_win_iso_trans_rot_2scales_50-50_7batches_RESULTSout10/2-1/model_fc-conv32-dc
set NET=D:\HolgerRoth\data\LymphNodes\MICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_RESULTS_train_and_crossvalid_6batch_random_10outputNodes\1-7\model_fc512-11pct-dc
set NET=D:\HolgerRoth\data\LymphNodes\MICCAI2014\Med\Med_LymphNodeData_LymphNodeData_highRes_win_iso_trans_rot_2scales_train_crossvalid_6batches_random_10outputNodes_RESULTS\1-7\model_fc512-11pct-dc
set NET=D:\HolgerRoth\data\LymphNodes\postMICCAI2014\Abd\LymphNodeData_highRes_win_iso_trans_rot_2scales_RESULTS\train_AxCoSa_balanced_and_cross_valid\fc512-11pct
set NET=D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_LINEAR_XYbetterCentered_RESULTS\train_AxRGB_balanced_6batches\fc512-11pct
set NET=D:\HolgerRoth\data\Spine\BoneLesions\BoneLesions_highRes_win_iso_trans_rot_4scales_BSPLINE_XYbetterCentered_RESULTS\train_AX_balanced_Gray_6batches\fc512-11pct
set NET=D:\HolgerRoth\data\Pancreas\rois\training_balanced_5batches_ConvNet\fc512-11pct

#6. Finally, substitute the real test set, and test on it using the procedure described here:
python convnet.py -f %NET% --multiview-test=1 --test-only=1 --logreg-name=logprob --test-range=6

# You should see output that looks something like this, in this case indicating a 10.9% test error.
======================Test output====================== (for ../data/cifar-10-py-colmajor_RESULTS/model_fc-11pct-dc)
logprob:  0.377645, 0.109200

# write predictions to file
python shownet.py -f %NET% --write-predictions=%NET%'_predictions' --test-range=6 --multiview-test=1

#7. Look at the trained net:
python shownet.py -f %NET% --show-cost=logprob

python shownet.py -f %NET% --show-cost=logprob --cost-idx=1

#7.1 show trained convolution filters:
python shownet.py -f %NET% --show-filters=conv1

python shownet.py -f %NET% --show-filters=conv1 --no-rgb=1

python shownet.py -f %NET% --show-filters=fc64 --channels=3

#8. To see the predictions that the net makes on test data, run the script like this:
python shownet.py -f %NET% --show-preds=probs

#9. To show only the mis-classified images, run the script like this:
python shownet.py -f %NET% --show-preds=probs --only-errors=1
