# Copyright (C) 2023, Computer Vision Lab, Seoul National University, https://cv.snu.ac.kr
#
# Copyright 2023 LucidDreamer Authors
#
# Computer Vision Lab, SNU, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from the Computer Vision Lab, SNU or
# its affiliates is strictly prohibited.
#
# For permission requests, please contact robot0321@snu.ac.kr, esw0116@snu.ac.kr, namhj28@gmail.com, jarin.lee@gmail.com.
import os
import numpy as np
import torch
import math
import cv2

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)





def generate_seed_360(viewangle, n_views, alternate=False):  #360,10
    N = n_views 
    render_poses = np.zeros((N, 3, 4))

    alternate_poses = lambda l: [l[0]] + [i for j in zip(l[1:len(l) // 2], l[-1:len(l) // 2:-1]) for i in j] + [
            l[len(l) // 2]]
    
    theta_list = []
    phi_list = []
    for i in range(N):
        th_degree = (viewangle/N)*i #degree
        theta_list.append(th_degree)
        phi_list.append(0)
        
    if alternate:
        theta_list = alternate_poses(theta_list)
    
    for i in range(N):
        th = math.radians(theta_list[i])    
        #Rotate around the Y-axis  left-and-right
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, -np.sin(th)], 
                                          [0, 1, 0], 
                                          [np.sin(th), 0, np.cos(th)]])
        
    return render_poses, theta_list, phi_list


def look_at(campos, opengl=False):

    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix

    target = np.zeros([3], dtype=np.float32)
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
        
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


def generate_seed_preset(n_views, phi=0):
    N = n_views 
    render_poses = np.zeros((N, 4, 4))
    theta_list = []
    phi_list =[]
    
    for i in range(N):
        th_degree = (360/N)*i  
        theta_list.append(th_degree)
        
        phi_list.append(phi)
        th = math.radians(th_degree)
        ph = math.radians(phi)
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th), 0, -np.sin(th)], 
                                                    [0, 1, 0], 
                                                    [np.sin(th), 0, np.cos(th)]]), 
                                        np.array([[1, 0, 0],
                                                    [0, np.cos(ph), -np.sin(ph)], 
                                                    [0, np.sin(ph), np.cos(ph)]]))
        render_poses[i,3,:] = [0.0, 0.0 ,0.0, 1.0]

    return render_poses, theta_list, phi_list

def gcd_pose_gs(th, ph):
    pose = np.zeros((4, 4))

    if th>180:
        th = th - 360

    # R_rotate = np.array([[-1, 0, 0],
    #                     [0, -1, 0],
    #                     [0, 0, 1]])

    th = math.radians(th)
    ph = math.radians(ph)
    
    x = np.cos(ph) * np.sin(th)
    y = - np.sin(ph)
    z = np.cos(ph) * np.cos(th)
    campos = np.array([x, y, z])
    R = look_at(campos)
    # pose[:3,:3] = np.dot(R, R_rotate)
    R[:, 1] *= -1
    pose[:3,:3] = R

    pose[3,:] = [0.0, 0.0 ,0.0, 1.0]

    return pose
def gcd_pose(th, ph):
    pose = np.zeros((3, 4))

    
    th = math.radians(360-th)
    ph = math.radians(ph)
    
    x = np.cos(ph) * np.sin(th)
    y = np.sin(ph)
    z = np.cos(ph) * np.cos(th)
    campos = np.array([x, y, z])
    R = look_at(campos)
    pose[:3,:3] = R
    
    return pose

def generate_pano360(n_views):
    N = n_views*3 + 1 + 1
    render_poses = np.zeros((N, 3, 4))
    render_poses_pc = np.zeros((N, 3, 4))
    theta_list = []
    phi_list = []
    
    _render_poses, _render_poses_pc, _theta_list, _phi_list = generate_360(n_views, phi=0)
    render_poses[0:n_views] =_render_poses
    render_poses_pc[0:n_views] =_render_poses_pc
    theta_list += _theta_list
    phi_list += _phi_list

    _render_poses, _render_poses_pc, _theta_list, _phi_list = generate_360(n_views, phi=-45)
    render_poses[n_views:n_views*2] =_render_poses
    render_poses_pc[n_views:n_views*2] =_render_poses_pc
    theta_list += _theta_list
    phi_list +=  [x for x in _phi_list]

    _render_poses, _render_poses_pc, _theta_list, _phi_list = generate_360(n_views, phi=45)
    render_poses[n_views*2:n_views*3] =_render_poses
    render_poses_pc[n_views*2:n_views*3] =_render_poses_pc
    theta_list += _theta_list
    phi_list += [x for x in _phi_list]
    
    polar_theta = [0,0]
    polar_phi = [-90,90]
    theta_list += polar_theta
    phi_list += polar_phi
    
    for i in range(len(polar_theta)):
        th = math.radians(polar_theta[i])
        ph = math.radians(polar_phi[i])
        
        x = np.cos(ph) * np.sin(th)
        y = np.sin(ph)
        z = np.cos(ph) * np.cos(th)

        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)

        [R1, _] = cv2.Rodrigues(y_axis * math.radians(360-polar_theta[i]))
        [R2, _] = cv2.Rodrigues(np.dot(R1, x_axis) * ph)

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)
        R = np.dot(R1,R2)
        
        render_poses[n_views*3 + i, :3, :3] = R
        render_poses_pc[n_views*3 + i, :3, :3] = R
    return render_poses, render_poses_pc, theta_list, phi_list
    
def generate_pano(viewangle, n_views, phi=0):  #360,10
    N = n_views 
    render_poses = np.zeros((N, 4, 4))
    theta_list = []
    phi_list =[]
    for i in range(N):
        th = (viewangle/N)*i  #degree
        theta_list.append(th)
        phi_list.append(phi)
        
        render_poses[i] = gcd_pose_gs(th, phi)
        
    return render_poses, theta_list, phi_list

def generate_finetune_pose(theta, r=2):
    render_pose = np.eye(4, dtype=np.float32)
    th = math.radians(theta)
    ph = math.radians(0)
    x = r * np.cos(ph) * np.sin(th)
    y = r * np.sin(ph)
    z = r * np.cos(ph) * np.cos(th)
    campos = np.array([x, y, z])
    R = look_at(campos)
    render_pose[:3,:3] = R
    render_pose[:3, 3] = campos
    return render_pose

def generate_360(n_views, phi=0):
    N = n_views 
    render_poses = np.zeros((N, 3, 4))
    render_poses_pc = np.zeros((N, 3, 4))
    
    theta_list = []
    phi_list = []
    for i in range(N):
        th_degree = (360/N)*i  #degree
        theta_list.append(th_degree)
        phi_list.append(phi)
        th = math.radians(th_degree)
        ph = math.radians(phi)

        x = np.cos(ph) * np.sin(th)
        y = np.sin(ph)
        z = np.cos(ph) * np.cos(th)
        campos = np.array([x, y, z])
        Rr = look_at(campos)

        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)

        [R1, _] = cv2.Rodrigues(y_axis * math.radians(360-th_degree))
        [R2, _] = cv2.Rodrigues(np.dot(R1, x_axis) * ph)

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)
        R = np.dot(R1,R2)
        render_poses[i, :3, :3] = R
        render_poses_pc[i, :3, :3] = Rr
        
    return render_poses, render_poses_pc, theta_list, phi_list


def generate_seed_360_half(viewangle, n_views): 
    N = n_views // 2
    halfangle = viewangle / 2
    render_poses = np.zeros((N*2, 3, 4))
    for i in range(N): 
        th = (halfangle/N)*i/180*np.pi
        render_poses[i,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    for i in range(N):
        th = -(halfangle/N)*i/180*np.pi
        render_poses[i+N,:3,:3] = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        render_poses[i+N,:3,3:4] = np.random.randn(3,1)*0.0 # Transition vector
    return render_poses






def generate_seed_hemisphere(center_depth, degree=5):
    degree = 5

    thlist = np.array([0])
    philist = np.array([0])

    # thlist = np.array([degree, 0, 0, 0, -degree])
    # philist = np.array([0, -degree, 0, degree, 0])
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        # curr_pose = np.zeros((1, 3, 4))
        d = center_depth # central point of (hemi)sphere / you can change this value
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
        # render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses





def generate_seed_lookdown():
    degsum = 60 
    thlist = np.concatenate((np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:], np.linspace(0, degsum, 4), np.linspace(0, -degsum, 4)[1:]))
    philist = np.concatenate((np.linspace(0,0,7), np.linspace(-22.5,-22.5,7)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], 
                                                    [0, 1, 0], 
                                                    [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), 
                                          np.array([[1, 0, 0],
                                                    [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], 
                                                    [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_back():
    movement = np.linspace(0, 10, 101)
    render_poses = [] # np.zeros((len(movement), 3, 4))
    for i in range(len(movement)):
        render_pose = np.zeros((3,4))
        render_pose[:3,:3] = np.eye(3)
        render_pose[:3,3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    movement = np.linspace(5, 0, 101)
    movement = movement[1:]
    for i in range(len(movement)):
        render_pose = np.zeros((3,4))
        render_pose[:3,:3] = np.eye(3)
        render_pose[:3,3:4] = np.array([[0], [0], [movement[i]]])
        render_poses.append(render_pose)

    return render_poses


def generate_seed_llff(degree, nviews, round=4, d=2.3):
    assert round%4==0
    thlist = degree * np.sin(np.linspace(0, 2*np.pi*round, nviews))
    philist = degree * np.cos(np.linspace(0, 2*np.pi*round, nviews))
    zlist = d/15 * np.sin(np.linspace(0, 2*np.pi*round//4, nviews))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        z = zlist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.array([d*np.sin(th/180*np.pi), 0, -z+d-d*np.cos(th/180*np.pi)]).reshape(3,1) + np.array([0, d*np.sin(phi/180*np.pi), -z+d-d*np.cos(phi/180*np.pi)]).reshape(3,1)# Transition vector
    return render_poses


def generate_seed_headbanging(maxdeg, nviews_per_round, round=3, fullround=1):
    radius = np.concatenate((np.linspace(0, maxdeg, nviews_per_round*round), maxdeg*np.ones(nviews_per_round*fullround), np.linspace(maxdeg, 0, nviews_per_round*round)))
    thlist  = 2.66*radius * np.sin(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    philist = radius * np.cos(np.linspace(0, 2*np.pi*(round+fullround+round), nviews_per_round*(round+fullround+round)))
    assert len(thlist) == len(philist)

    render_poses = np.zeros((len(thlist), 3, 4))
    for i in range(len(thlist)):
        th = thlist[i]
        phi = philist[i]
        
        render_poses[i,:3,:3] = np.matmul(np.array([[np.cos(th/180*np.pi), 0, -np.sin(th/180*np.pi)], [0, 1, 0], [np.sin(th/180*np.pi), 0, np.cos(th/180*np.pi)]]), np.array([[1, 0, 0], [0, np.cos(phi/180*np.pi), -np.sin(phi/180*np.pi)], [0, np.sin(phi/180*np.pi), np.cos(phi/180*np.pi)]]))
        render_poses[i,:3,3:4] = np.zeros((3,1))

    return render_poses


def generate_seed_around(viewangle, n_views, alternate=True, phi_tree=[0,-45,45]):
    
    alternate_poses = lambda l: [l[0]] + [i for j in zip(l[1:len(l) // 2], l[-1:len(l) // 2:-1]) for i in j] + [
        l[len(l) // 2]]
    
    
    n_phi = len(phi_tree)
    N = n_views * n_phi
    theta_tmp_list = []
    theta_list = []
    phi_list = []

    render_poses = np.zeros((N, 3, 4))
    render_relative_poses = np.zeros((N, 3, 4))
    
    for i in range(n_views):
        th_degree = (viewangle/n_views)*i 
        theta_tmp_list.append(th_degree)
        
    if alternate:
        theta_tmp_list = alternate_poses(theta_tmp_list)
    
    for i in range(n_views):
        th_degree = theta_tmp_list[i]  #degree
        
        for j in range(n_phi):
            
            phi = phi_tree[j]
            phi_list.append(phi)
            theta_list.append(th_degree)
            
            th = math.radians(th_degree)
            ph = math.radians(phi)

            x = np.cos(ph) * np.sin(th)
            y = np.sin(ph)
            z = np.cos(ph) * np.cos(th)
            campos = np.array([x, y, z])
            Rr = look_at(campos)

            x_axis = np.array([1.0, 0.0, 0.0], np.float32)
            y_axis = np.array([0.0, 1.0, 0.0], np.float32)

            [R1, _] = cv2.Rodrigues(y_axis * math.radians(360-th_degree))
            [R2, _] = cv2.Rodrigues(np.dot(R1, x_axis) * ph)

            R1 = np.linalg.inv(R1)
            R2 = np.linalg.inv(R2)
            R = np.dot(R1,R2)
            # print(f"R_phi-[{th_degree}-{phi}]", R_phi)
            # print("================================")
            

            # R = np.matmul(np.array([[np.cos(th), 0, -np.sin(th)], 
            #                         [0, 1, 0], 
            #                         [np.sin(th), 0, np.cos(th)]]), 
            #               np.array([[1, 0, 0],
            #                         [0, np.cos(phi), -np.sin(phi)], 
            #                         [0, np.sin(phi), np.cos(phi)]]))
    
            render_poses[i*n_phi+j, :3, :3] = R
            render_relative_poses[i*n_phi+j, :3, :3] = Rr
            
    return render_poses, render_relative_poses, theta_list, phi_list




def get_camerapaths():
    preset_json = {}
    for cam_path in ["back_and_forth", "llff", "headbanging"]:
        if cam_path == 'back_and_forth':
            render_poses = generate_seed_back()
        elif cam_path == 'llff':
            render_poses = generate_seed_llff(5, 400, round=4, d=2)
        elif cam_path == 'headbanging':
            render_poses = generate_seed_headbanging(maxdeg=15, nviews_per_round=180, round=2, fullround=0)
        else:
            raise("Unknown pass")
            
        yz_reverse = np.array([[1,0,0], [0,-1,0], [0,0,-1]])
        blender_train_json = {"frames": []}
        for render_pose in render_poses:
            curr_frame = {}
            ### Transform world to pixel
            Rw2i = render_pose[:3,:3]
            Tw2i = render_pose[:3,3:4]

            # Transfrom cam2 to world + change sign of yz axis
            Ri2w = np.matmul(yz_reverse, Rw2i).T
            Ti2w = -np.matmul(Ri2w, np.matmul(yz_reverse, Tw2i))
            Pc2w = np.concatenate((Ri2w, Ti2w), axis=1)
            Pc2w = np.concatenate((Pc2w, np.array([0,0,0,1]).reshape((1,4))), axis=0)

            curr_frame["transform_matrix"] = Pc2w.tolist()
            blender_train_json["frames"].append(curr_frame)

        preset_json[cam_path] = blender_train_json

    return preset_json


def get_pcdGenPoses(pcdgenpath, argdict={}):
    if pcdgenpath == 'rotate360':
        render_poses = generate_seed_360(360, argdict['n_views'], argdict['alternate'])
    elif pcdgenpath == 'lookaround':
        render_poses = generate_seed_around(360, 10, argdict['alternate'], argdict['phi_tree'])
    elif pcdgenpath == 'pano360':
        render_poses = generate_pano360(argdict['n_views'])
    elif pcdgenpath == 'hemisphere':
        render_poses = generate_seed_hemisphere(argdict['center_depth'])
    elif pcdgenpath == 'pers2pano':
        render_poses = generate_pano(360, argdict['n_views'], argdict['phi'])
    else:
        raise("Invalid pcdgenpath")
    return render_poses


