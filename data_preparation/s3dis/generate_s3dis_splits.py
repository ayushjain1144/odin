import numpy as np
import os


if __name__=="__main__":
    ROOT_PATH = '/projects/katefgroup/language_grounding/s3dis_frames'
    all_areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5', 'area_6']
    
    val_areas = ['area_1']
    train_areas = [area for area in all_areas if area not in val_areas]
    
    train_list = []
    val_list = []
    
    for folder in os.listdir(ROOT_PATH):
        area_prefix = folder[:6]
        if area_prefix in train_areas:
            train_list.append(folder)
        elif area_prefix in val_areas:
            val_list.append(folder)
            
    train_eval = np.random.choice(train_list, 10, replace=False)
    debug_split = np.random.choice(train_eval, 2, replace=False) 
    
    
    data_path = 's3dis_splits'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # write all the splits to a file
    train_file_path = f'{data_path}/s3dis_{val_areas[0]}_train.txt'
    val_file_path = f'{data_path}/s3dis_{val_areas[0]}_val.txt'
    debug_file_path = f'{data_path}/two_{val_areas[0]}_scene.txt'
    train_eval_file_path = f'{data_path}/ten_{val_areas[0]}_scene.txt'
    
    with open(train_file_path, 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    
    with open(val_file_path, 'w') as f:
        for item in val_list:
            f.write("%s\n" % item)
            
    with open(debug_file_path, 'w') as f:
        for item in debug_split:
            f.write("%s\n" % item)
    
    with open(train_eval_file_path, 'w') as f:
        for item in train_eval:
            f.write("%s\n" % item)
    