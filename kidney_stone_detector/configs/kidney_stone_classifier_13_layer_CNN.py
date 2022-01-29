import torch

config = {
	'cliplow': -200,
	'cliphigh' : 1000,
	'model':  '13layerCNN',
	'data_root': "/home/delton/data/kidney_stone_boxes_4",
 	'box_size': 24,
	'in_channels': 1,  # just leave this as it is
	'n_classes': 1,
	'batch_size': 8,
	'initial_LR': 0.0005,
	'factor_LR': 1,
	'n_augmented_versions' : 32, #sets ratio
	'patience': 100,  # number of epochs with no improvement before lr is decreased
	'step_factor_LR': 0.5,  # once learning stagnates, the learning rate will be decreased by this factor
	'validation_every_n': 500,  # check the model performance on validation and test data every n iterations
	'HUoffset': 0, # -1024,  # for inference only. should be 0 unless for CTC dataset -- literally no clue what this does
	'REMARKS':'please be aware of incorrect HU in CTC files. HUoffset has to be set to -1024 for these cases for inference'
	}
