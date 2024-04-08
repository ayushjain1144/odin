import json
from glob import glob
import os
import shutil
import imageio
from PIL import Image
import ipdb
import numpy as np
from natsort import natsorted

import ipdb
st = ipdb.set_trace


def get_index( color ):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[:, :, 0] * 256 * 256 + color[:, :, 1] * 256 + color[:, :, 2]

""" Label functions """
def load_labels( label_file ):
    """ Convenience function for loading JSON labels """
    with open( label_file ) as f:
        return json.load( f )

def parse_label( label ):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split( "_" )
    res[ 'instance_class' ] = clazz
    res[ 'instance_num' ] = int( instance_num )
    res[ 'room_type' ] = room_type
    res[ 'room_num' ] = int( room_num )
    res[ 'area_num' ] = int( area_num )
    return res


def eul2rom(eul):
    phi = eul[0] 
    theta = eul[1]
    psi = eul[2]
    ax = np.array([
        [1,0,0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]])
    ay = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0,1,0],
        [-np.sin(theta), 0, np.cos(theta)]])
    az = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0,0,1]])
    A = np.matmul(np.matmul(az, ay), ax)
    return A

def process_pose(frame_id, pose_file, folder_processed_pose, area):
    with open(pose_file, 'r') as f:
        data = json.load(f)

    # intrinsics
    camera_k_matrix = data['camera_k_matrix'] # projection matrix
    intrinsics = np.array(camera_k_matrix).tolist() # 3x3 homogen array 
    
    # # extrinsics
    # A = data['final_camera_rotation']
    # from scipy.spatial.transform import Rotation as R
    # # import ipdb; ipdb.set_trace()
    # A_homo = np.eye(4)
    # A = R.from_euler('xyz', A).as_matrix()
    # A_homo[:3, :3] = A

    # R_homo = np.eye(4)
    # R_ = np.array(data['camera_rt_matrix'])[:3, :3]
    # R_homo[:3, :3] = R_

    # c_mat = np.eye(4)
    # c = data['camera_location']
    # c_mat[:3, 3] = c
    # pose = c_mat @ (A_homo @ R_homo @ A_homo)

    camera_rt_matrix = data["camera_rt_matrix"] # camera pose
    camera_rt_matrix.append([0, 0, 0, 1]) # 4x4 homogen array
    camera_rt_matrix = np.linalg.inv(np.array(camera_rt_matrix))
    # if not np.allclose(pose, camera_rt_matrix):
    #     st()
    
    if 'area_5b' in area:
        print("fixing")
        camera_rt_matrix = np.array([
            [0, 1, 0, -4.10],
            [-1, 0, 0, 6.25],
            [0, 0, 1, 0.0],
            [0, 0, 0, 1]
        ]) @ camera_rt_matrix
    
    camera_rt_matrix = camera_rt_matrix.tolist()
    
    room = data['room']
    # if ('area_2' in area and 'hallway_11' in room) or ('area_5' in area and 'hallway_6' in room):
        

    file = open(folder_processed_pose + '/' + frame_id + '.txt','w')
    for item in camera_rt_matrix:
        for i in range(len(item)):
            file.write(str(item[i])+" ")
        file.write("\n")
    
    folder_processed_intrinsic = folder_processed_pose.replace('pose', 'intrinsic')
    file = open(folder_processed_intrinsic + '/' + frame_id + '.txt','w')
    for item in intrinsics:
        for i in range(len(item)):
            file.write(str(item[i])+" ")
        file.write("\n")

def process_color(frame_id, file, folder_processed):
    image = Image.open(file)
    image = image.resize((256, 256), resample=Image.BILINEAR)
    image.save(os.path.join(folder_processed, frame_id + '.png'))
    # also save the original image:
    folder_processed = folder_processed.replace('color', 'color_original')
    shutil.copyfile(file, os.path.join(folder_processed, frame_id + '.png'))

def process_depth(frame_id, file, folder_processed):
    numpy_depth = np.array(Image.open(file), dtype=np.uint16)
    numpy_depth = numpy_depth.astype(np.float32) / 512.0
    numpy_depth[numpy_depth > 126] = 0.0
    # if numpy_depth.max() > 50:
    #     st()
    numpy_depth[numpy_depth > 50] = 0.0
    numpy_depth = (numpy_depth * 1000.0).astype(np.uint16)
    image_highres = Image.fromarray(numpy_depth)
    image = Image.fromarray(numpy_depth)
    
    # image = Image.fromarray(np.array(Image.open(file), dtype=np.uint16))
    image = image.resize((256, 256), resample=Image.NEAREST)
    image.save(os.path.join(folder_processed, frame_id + '.png'))
    # also save the original image:
    folder_processed = folder_processed.replace('depth', 'depth_original')
    image_highres.save(os.path.join(folder_processed, frame_id + '.png'))
    # shutil.copyfile(file, os.path.join(folder_processed, frame_id + '.png'))

def process_labels(frame_id, file, folder_processed):
    _, _, _, _, _, frame_id, _, _ = file.split('/')[-1].split('_')
    #img = imageio.imread(file)
    #img = img.resize(IMAGE_SIZE)
    image = Image.open(file)
    image = np.array(image.resize((256, 256), resample=Image.BILINEAR))
    idx = get_index(image)
    # instance_label = labels[idx[3,3]]
    # instance_label_as_dict = parse_label(instance_label)
    # print(instance_label_as_dict)
    imageio.imwrite(folder_processed + '/' + frame_id + '.png', idx)
    folder_processed = folder_processed.replace('instance', 'instance_original')
    shutil.copyfile(file, os.path.join(folder_processed, frame_id + '.png'))

def process_scene(area_name, scene_name, files, labels):

    folder_name = area_name + '_' + scene_name
    if 'area_5' in folder_name:
        folder_name = folder_name.replace(area_name, 'area_5')
    print("processing scene: ", folder_name)
 
    folder_processed_depth = SAVE_ROOT + folder_name + '/depth'
    os.makedirs(folder_processed_depth, exist_ok=True)
    folder_processed_depth_original = SAVE_ROOT + folder_name + '/depth_original'
    os.makedirs(folder_processed_depth_original, exist_ok=True)

    folder_processed_color = SAVE_ROOT + folder_name + '/color'
    os.makedirs(folder_processed_color, exist_ok=True)
    folder_processed_color_original = SAVE_ROOT + folder_name + '/color_original'
    os.makedirs(folder_processed_color_original, exist_ok=True)

    folder_processed_pose = SAVE_ROOT + folder_name + '/pose'
    os.makedirs(folder_processed_pose, exist_ok=True)
    
    folder_processed_intrinsic = SAVE_ROOT + folder_name + '/intrinsic'
    os.makedirs(folder_processed_intrinsic, exist_ok=True)

    folder_processed_instance = SAVE_ROOT + folder_name + '/instance'
    os.makedirs(folder_processed_instance, exist_ok=True)
    folder_processed_instance_original = SAVE_ROOT + folder_name + '/instance_original'
    os.makedirs(folder_processed_instance_original, exist_ok=True)
    
    assert len(files['color_files']) == len(files['pose_files']) == len(files['depth_files']) == len(files['label_files'])

    for i, (rgb_file, pose_file, depth_file, label_file) in enumerate(zip(files['color_files'], files['pose_files'], files['depth_files'], files['label_files'])):
        _, _, _, _, _, frame_id, _, _ = rgb_file.split('/')[-1].split('_')
        # process_color(str(i), rgb_file, folder_processed_color)
        # process_depth(str(i), depth_file, folder_processed_depth)
        # process_labels(str(i), label_file, folder_processed_instance)
        process_pose(str(i), pose_file, folder_processed_pose, area_name)

def parse_scenes(area):
    scenes = {}

    # if 'area_5' in area:
    #     area_5a = area.replace('area_5b', 'area_5a')
    #     area_5b = area.replace('area_5', 'area_5b')
    #     files = glob(area_5a + '/data/pose/*.json')
    #     files += glob(area_5b + '/data/pose/*.json')
        
    files = glob(area + '/data/pose/*.json')

    # collect rooms
    for file in natsorted(files):
        _, cam_id, roomtype, room_id, _, frame_id, _, _ = file.split('/')[-1].split('_')
        room = roomtype + '_' + room_id

        pose_file = file
        file = file.replace('json', 'png')
        depth_file = file.replace('pose', 'depth')
        rgb_file = file.replace('pose', 'rgb')
        semantic_file = file.replace('pose', 'semantic')
        if not (os.path.isfile(depth_file) and 
                os.path.isfile(rgb_file) and 
                #os.path.isfile(normal_file) and 
                os.path.isfile(semantic_file)):
            print("File not found: ", depth_file)
            ipdb.set_trace()

        if room in scenes:
            scenes[room]['pose_files'].append(pose_file)
            scenes[room]['depth_files'].append(depth_file)
            scenes[room]['color_files'].append(rgb_file)
            scenes[room]['label_files'].append(semantic_file)
            # scenes[room]['instance_files'].append(instance_files)
        else:
            scenes[room] = {
                'depth_files': [depth_file],
                'color_files': [rgb_file],
                'pose_files': [pose_file],
                'label_files': [semantic_file],
                # 'instance_files': []
            }
    return scenes

if __name__ == '__main__':

    labels = load_labels('/projects/katefgroup/language_grounding/s3dis/semantic_labels.json')
    
    SAVE_ROOT = '/projects/katefgroup/language_grounding/s3dis_frames/'
    # areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b', 'area_6']
    areas = ['area_5a', 'area_5b']
    # areas = ['area_5']
    
    scenes_area_5a = parse_scenes('/projects/katefgroup/language_grounding/s3dis/area_5a')
    scenes_area_5b = parse_scenes('/projects/katefgroup/language_grounding/s3dis/area_5b')
    
    merge_scenes = [scene for scene in scenes_area_5a if scene in scenes_area_5b]
    
    # st()
    process_scene('area_5a', 'office_20', scenes_area_5a['office_20'], labels=labels)
    
    # for area_name in areas:
    #     print("Processing area: ", area_name)
    #     area = f'/projects/katefgroup/language_grounding/s3dis/{area_name}'

    #     scenes = parse_scenes(area)
    #     st()
    #     for scene_name in scenes:
    #         process_scene(area_name, scene_name, scenes[scene_name], labels=labels)