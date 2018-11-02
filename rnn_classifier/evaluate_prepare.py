import sys
from myconfig import *
sys.path.append(diva_util_path)
from diva_util import *
import json
import glob

vidlist = valid_vid_list
# vidlist = ['VIRAT_S_040103_01_000132_000195']

# vidlist = ['VIRAT_S_040103_00_000000_000120', 'VIRAT_S_040103_01_000132_000195', 'VIRAT_S_040103_02_000199_000279', 
#            'VIRAT_S_040103_03_000284_000425', 'VIRAT_S_040103_06_000836_000909', 'VIRAT_S_040104_07_001268_001348',
#            'VIRAT_S_000007', 'VIRAT_S_000008']


with open("frame_nums.txt", "r") as ins:
    array = []
    for line in ins:
        line = line.rstrip('\n')
        line = line.split(" ")
        array.append(line)
array = np.asarray(array)

vid_dict = {}
for ix in range(array.shape[0]):
    vid_dict[array[ix][0]] = array[ix][1]
    
vfn_index = {}
for vid in vidlist:
#     img_fn_list = glob.glob(img_path+vid+'/*.jpg')
    try:
        start,end = str(1), str(vid_dict[vid])
        print("present!")
    except:
        print("Not present anywhere!")
    
    vindex = {}
    vindex['framerate'] = 30.0
    vindex['selected'] = {start:1, end:0}
    vfn_index[vid+'.mp4'] = vindex
#json.dump(vfn_index, open('test/validation_file-index.json','w'), encoding='ascii', indent=4)
f = open('test/validation_file-index.json','w')
f.write(json.dumps(vfn_index, indent=4))

ref = {}
ref['activities'] = []
actcount = 1
for vid in vidlist:
    actfn = valid_annot_path+vid+'.activities.yml'
    try:
        actlist = parse_diva_act_yaml(actfn)
        for act in actlist:
            if 'meta' in act.keys(): continue
            if act['act2'] not in ACT_NAMES_V1: continue
            act_ref = {}
            act_ref['activity'] = act['act2']
            act_ref['activityID'] = actcount
            act_ref['localization'] = {}
            start, end = act['timespan'][0]['tsr0']
            act_ref['localization'][vid+'.mp4'] = {str(start):1, str(end):0}
            actcount+=1
            ref['activities'].append(act_ref)
    except:
        print("Not present anywhere!")
    
#     for act in actlist:
#         if 'meta' in act.keys(): continue
#         if act['act2'] not in ACT_NAMES_V1: continue
#         act_ref = {}
#         act_ref['activity'] = act['act2']
#         act_ref['activityID'] = actcount
#         act_ref['localization'] = {}
#         start, end = act['timespan'][0]['tsr0']
#         act_ref['localization'][vid+'.mp4'] = {str(start):1, str(end):0}
#         actcount+=1
#         ref['activities'].append(act_ref)
ref['filesProcessed'] = [vid+'.mp4' for vid in vidlist]
json.dump(ref, open('test/validation.json','w'), encoding='ascii', indent=4)

act_dict = {}
for act in ACT_NAMES_V1:
    if 'vehicle' not in act: continue
    act_dict[act] = {}
json.dump(act_dict, open('test/validation_activity-index.json','w'),encoding='ascii', indent=4)
