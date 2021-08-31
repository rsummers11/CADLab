import torch

config = {
	'model' :  'model.MultiData_big_lowfilter_stretch_old',
	'taskstouse' : (3,),
	'originalXY' : 256 ,
	'originalZ' : 192,
	'in_channels' : 1,
	'n_classes' : 1,
	'augmentversions' : 8,
	'batch_size' : torch.cuda.device_count(),
	'initial_LR' : 0.0001,
	'factor_LR' : 1,
	'patience' : 30,
	'step_factor_LR' : 0.8,
	'n_filter_per_level' :(32, 64, 128, 256, 512),
	'number_train' : None,
	'validation_every_n':100,
	'deformfactor' : 8,
	'material' : False,
	'HUoffset': 0,
	'REMARKS': 'please be aware of incorrect HU in CTC files. HUoffset has to be set to -1024 for these cases'
	}
