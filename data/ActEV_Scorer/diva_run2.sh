#!/usr/bin/env bash
#diva_path='/home/tingyaoh/research/diva/action_proposal'
diva_path='/home/tingyaoh/test_diva'

python2 ActEV_Scorer.py \
	ActEV18_AD \
	-s $diva_path/validation_sysout_3_layer_ld_0.05.json \
	-r $diva_path/validation.json \
	-a $diva_path/validation_activity-index.json \
	-f $diva_path/validation_file-index.json \
	-o diva_run-output \
	-v 
    
