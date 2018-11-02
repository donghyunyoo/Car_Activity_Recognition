import sys
import os.path as osp
from myconfig import *
sys.path.append(diva_util_path)
from diva_util import *
from collections import defaultdict
import numpy as np
from action_proposal_util import Vehicle
import pickle
# from orientation import get_model, extract_orientation
import json


"""
functions for 3d pose
"""
def read_3d_pose(vid, pose_root):
#     print("vid",vid)
#     print("pose_root", pose_root)
    json_fn_list = sorted(glob.glob(os.path.join(pose_root, vid)+'/*.json'))
    jobj_list = [json.load(open(fn)) for fn in json_fn_list]
    print(len(jobj_list))
    return jobj_list

def find_3d_poses(bbox, jsonobj):
    max_iou, max_idx = -1, -1
    for idx, obj in enumerate(jsonobj['objects']):
        obj_bbox = obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']
        iou_value = iou(bbox, obj_bbox)
        if iou_value>max_iou: 
            max_idx = idx
            max_iou = iou_value
#             print("max_iou", max_iou)
    if max_idx>-1 and max_iou>0.30:
#         print('rotation:',jsonobj['objects'][max_idx]['rotation'])
#         print("max_idx", max_idx)
#         print("max_iou", max_iou)
#         print(type(jsonobj['objects'][max_idx]['rotation']))
        rot = jsonobj['objects'][max_idx]['rotation']
        rot = rot/360.0
        return rot
    else: 
        return -1

# def find_3d_poses(bbox, jsonobj):
#     max_iou, max_idx = -1, -1
#     for idx, obj in enumerate(jsonobj['objects']):
#         obj_bbox = obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']
#         iou_value = iou(bbox, obj_bbox)
# #         print("iou_value", max(iou_value))
#         if iou_value>max_iou: max_idx = idx
#     if max_idx>-1 and iou_value>0.85:
# #         print("iou_value", iou_value)
#         print('rotation:',jsonobj['objects'][max_idx]['rotation'])
#         return jsonobj['objects'][max_idx]['rotation']
#     else: 
#         return -1

# tracklet: list of (t, bbox)
def get_time_bbox_dict(tracklet):
    bbox_dict = defaultdict()
    for t, box in tracklet:
        bbox_dict[t] = box
    return bbox_dict

def xywh2xxyy(box):
    x,y,w,h = box
    return x,y,x+w,y+h

"""
vehicle feature functions
    tracklet: list of (t, bbox)
"""
def velocity_estimation(tracklet, dist=3):
    vlst = []
    for i in range(len(tracklet)):
        t, box = tracklet[i]
        if i+dist>=len(tracklet): break
        v = v_estimate(t, box, tracklet[i+dist][0], tracklet[i+dist][1])
        v = np.linalg.norm(v)
        vlst.append(v)
    if len(vlst)==0: return 0
    return np.mean(vlst)
        
def v_estimate(t1, box1, t2, box2):
    x1, y1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    x2, y2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    v = [(x2-x1), (y2-y1)]
    return v

def angle_velocity(tracklet, dist=5):
    wlst = []
    for i in range(len(tracklet)):
        t, box = tracklet[i]
        if i+2*dist>=len(tracklet): break
        w = w_estimate(t, box, tracklet[i+dist][0], tracklet[i+dist][1], tracklet[i+2*dist][0], tracklet[i+2*dist][1])
        wlst.append(w)
    if len(wlst)==0: return 0
    return np.mean(wlst)
    
def w_estimate(t1,box1,t2,box2,t3,box3):
    x1, y1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    x2, y2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    x3, y3 = (box3[0]+box3[2])/2, (box3[1]+box3[3])/2
    v1,v2 = [x2-x1,y2-y1], [x3-x2,y3-y2]
    cos = ((x2-x1)*(x3-x2)+(y2-y1)*(y3-y2))/(np.linalg.norm(v1)*np.linalg.norm(v2))
    t_1, t_2 = (t1+t2)/2, (t2+t3)/2
    if np.isnan(cos): return 0
    cos = max(min(1,cos),-1)
    if cos<0: return (3.1416-np.arccos(cos))/(t_2-t_1)
    return np.arccos(cos)/(t_2-t_1)

#model = get_model('resnet18', 'resnet18_pretrained_16b_flip_rot_acc70v0.pt')
print('after loading model')
seglen = 60

#for vid in train_vid_list+valid_vid_list:
# scene_2 = ['VIRAT_S_040103_00_000000_000120', 'VIRAT_S_040103_01_000132_000195', 'VIRAT_S_040103_02_000199_000279', 
#           'VIRAT_S_040103_03_000284_000425', 'VIRAT_S_040103_05_000729_000804', 'VIRAT_S_040103_06_000836_000909',
#           'VIRAT_S_040103_07_001011_001093', 'VIRAT_S_040103_08_001475_001512', 'VIRAT_S_040104_00_000120_000224',
#           'VIRAT_S_040104_01_000227_000457', 'VIRAT_S_040104_02_000459_000721', 'VIRAT_S_040104_04_000854_000934',
#           'VIRAT_S_040104_05_000939_001116', 'VIRAT_S_040104_06_001121_001241', 'VIRAT_S_040104_07_001268_001348',
#           'VIRAT_S_040104_08_001353_001470', 'VIRAT_S_040104_09_001475_001583']
scene_2 = ['VIRAT_S_040103_00_000000_000120', 'VIRAT_S_040103_01_000132_000195', 'VIRAT_S_040103_02_000199_000279', 
          'VIRAT_S_040103_03_000284_000425', 'VIRAT_S_040103_05_000729_000804', 'VIRAT_S_040103_06_000836_000909',
          'VIRAT_S_040103_07_001011_001093', 'VIRAT_S_040103_08_001475_001512', 'VIRAT_S_040104_00_000120_000224',
          'VIRAT_S_040104_01_000227_000457', 'VIRAT_S_040104_02_000459_000721', 'VIRAT_S_040104_04_000854_000934',
          'VIRAT_S_040104_05_000939_001116', 'VIRAT_S_040104_06_001121_001241', 'VIRAT_S_040104_07_001268_001348',
          'VIRAT_S_040104_09_001475_001583']

scene_1 = ['VIRAT_S_000000','VIRAT_S_000001','VIRAT_S_000002', 'VIRAT_S_000005','VIRAT_S_000007', 'VIRAT_S_000008']
# for vid in ['VIRAT_S_040103_01_000132_000195']:
for vid in scene_1 + scene_2:
#     print("vid", vid)
    featlst = []
    lablst = []
    
    ### new function: read 3d poses from json files
#     poses_list = read_3d_pose(vid, '3d_pose/')
    poses_list = read_3d_pose(vid, '3d_pose/')
#     print("poses list len", poses_list[0].keys())

    if vid in train_vid_list:
        print("vid", vid)
        print("present in train_vid_list")
        actlst = parse_diva_act_yaml(train_annot_path+vid+'.activities.yml')
        geomlst = parse_diva_geom_yaml(train_annot_path+vid+'.geom.yml')
        geom_id_dict = get_geom_id_list(geomlst)
        typedict = parse_diva_type_yaml(train_annot_path+vid+'.types.yml')
    elif vid in valid_vid_list:
        print("present in valid_vid_list")
        actlst = parse_diva_act_yaml(valid_annot_path+vid+'.activities.yml')
#         print(actlst)
        geomlst = parse_diva_geom_yaml(valid_annot_path+vid+'.geom.yml')
#         print(geomlst)
        geom_id_dict = get_geom_id_list(geomlst)
#         print(geom_id_dict)
        typedict = parse_diva_type_yaml(valid_annot_path+vid+'.types.yml')
#         print(typedict)
    else:
        print("Not present anywhere!")

    ### extract all ground truth tracklet
    
    act_tubes = []
    act_idxs = []
    for act in actlst:
        if 'meta' in act.keys(): continue
        if act['act2'] not in VEHICLE_ACT_NAMES: continue
        span, bb_lst = get_act_tubelet(act, geom_id_dict)
        actlab = VEHICLE_ACT_NAMES.index(act['act2'])+1
        act_tubes.append((actlab, span, bb_lst))
        act_idxs.append(actlab)
    matched = [0]*len(act_tubes)

    ### extract detection result
    try:
        trackdict = read_mot_as_defaultdict(car_trk_dir+vid+'.txt')
    except:
        print("Not present anywhere!")
    for k in trackdict.keys():
#         print("k",k)
        start = min([t for t, box in trackdict[k]])
        end = max([t for t, box in trackdict[k]])
        if end-start<10: continue
        
        labs = np.zeros((end-start+1,))
        bbox_dict = get_time_bbox_dict(trackdict[k])
        for j,act_tube in enumerate(act_tubes):
            actlab, span, bb_lst = act_tube
            s2,e2 = span
            if tiou(start,end,s2,e2)>0: 
                    iou_values = []
                    for tt in range(max(start,s2),min(end,e2)+1):
                        try:
                            iou_values.append(iou(xywh2xxyy(bbox_dict[tt]), bb_lst[tt-s2]))
                        except KeyError:
                            continue
                            print('tracker missing')
                    if np.mean(iou_values)>0.8: 
                        #print('matched!',k, max(start,s2),min(end,e2))
                        s, e = max(start,s2),min(end,e2)
                        labs[s-start:e-start+1] = actlab
                        matched[j] = 1

        #if np.sum(labs)!=0: print(labs)
        car = Vehicle(k, trackdict[k])

        ### extract 3-d pose
        degreelist = []
        for t in range(start, end+1):
            
#             print("bbox", bbox)
#             print("t",t)
            try:
                bbox = car.get_box_from_t(t)
                degree = find_3d_poses(bbox, poses_list[t])
                degreelist.append(degree)
            except:
                print("IndexError: list index out of range!")
            
        # Post processing of outputs
        for ix in range(len(degreelist)):
            if degreelist[ix] == -1:
                ctr = ix - 1
                while(True):
                    if ctr == -1:
                        break
                    elif degreelist[ctr] != -1:
                        degreelist[ix] == degreelist[ctr]
                        break
                    else:
                        ctr = ctr - 1         
                       
                
        vmean = np.mean([np.linalg.norm(car.vs[i,:]) for i in range(len(car.vs))])
        trk_lab = 1 if np.sum(labs)>0 else 0
        #print('average v:',vmean, trk_lab)
        if vmean<1 and trk_lab==1: print(('average v:',vmean, trk_lab))
        if vmean>=1:
            ## TODO: here you need to concatenate degreelist and car.vs
            ##
            ##
            try:
#                 print("Shape of degreelist", len(degreelist))
                print("Shape of car.vs", car.vs.shape)
                print("Shape of car.cen", car.cen.shape)
#                 degreelist = np.asarray(degreelist)
#                 print(degreelist)
#                 degreelist = np.reshape(degreelist, (degreelist.shape[0],1))
#                 print("Shape of degreelist now", degreelist.shape)
#                 final_features = np.hstack((car.vs,degreelist))
                # Appending centers to the velocities
                final_features = np.hstack((car.vs, car.cen))
#                 featlst.append(car.vs)
                print("Shape of final features now", final_features.shape)
                featlst.append(final_features)
                lablst.append(labs)
            except:
                pass
    
#     print featlst
    
    print(vid, matched,act_idxs)
    pickle.dump((featlst,lablst), open('data/'+vid+'.pkl','wb'))
