#!/usr/bin/env bash
#diva_path='/home/tingyaoh/research/diva/action_proposal'
diva_path='/home/rishi/Vehice_Action_Classifier/src/rnn_classifier'

python2 ActEV_Scorer.py \
	ActEV18_AD \
	-s $diva_path/test/validation_sysout.json \
	-r $diva_path/test/validation.json \
	-a $diva_path/test/validation_activity-index.json \
	-f $diva_path/test/validation_file-index.json \
	-o diva_run-output \
	-v 
    
