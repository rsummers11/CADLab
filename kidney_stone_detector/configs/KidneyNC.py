import torch

config = {
	'model' :  'model.MultiData_big_lowfilter_stretch_old',
	'taskstouse' : (13,),
	'testsplits' : 20, # final model no big test set needed
	'originalXY' : 256 ,
	'originalZ' : 192,
	'in_channels' : 1,
	'n_classes' : 1,
	'augmentversions' : 8,
    'cliplow' :  -500,
    'cliphigh' : 2000,
	'batch_size' : 1,
	'initial_LR' : 0.0002,
	'factor_LR' : 1,
	'patience' : 20,
	'step_factor_LR' : 0.5,
	'n_filter_per_level' :(32, 64, 128, 256, 512),
	'number_train' : None,
	'validation_every_n':100,
	'deformfactor' : 8,
	'material' : False,
	'HUoffset': 0,
	'REMARKS': 'please be aware of incorrect HU in CTC files. HUoffset has to be set to -1024 for these cases for inference'
	}
