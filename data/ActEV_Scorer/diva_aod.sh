#!/usr/bin/env bash
#diva_path='/home/tingyaoh/research/diva/action_proposal'
diva_path='/home/tingyaoh/research/diva/vehicle_events'
gt_path='/home/tingyaoh/research/diva/test/VIRAT-V1_JSON_test-indices_drop4_20180425/'
python2 ActEV_Scorer.py \
	ActEV18_AOD \
	-s $diva_path/test/test_sys_spatial_out1.json \
	-a $gt_path/activity-index.json \
	-f $gt_path/file-index.json \
	-V
    
