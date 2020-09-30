# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import tensorflow as tf
import glob
import mayavi.mlab as mlab
from visualization_utils import *
import mayavi.mlab as mlab
from grasp_data_reader import regularize_pc_point_count
import rospy
from geometry_msgs.msg import Pose, PoseArray,Point, PointStamped, Vector3
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from panda_simulation.srv import computeGrasps
#import ros_numpy.point_cloud2 as pc2
import tf as tf_ros
import ros_numpy


def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--vae_checkpoint_folder',
        type=str, 
        default='checkpoints/latent_size_2_ngpus_1_gan_1_confidence_weight_0.1_npoints_1024_num_grasps_per_object_256_train_evaluator_0_')
    parser.add_argument(
        '--evaluator_checkpoint_folder', 
        type=str, 
        default='checkpoints/npoints_1024_train_evaluator_1_allowed_categories__ngpus_8_/'
    )
    parser.add_argument(
        '--gradient_based_refinement',
        action='store_true',
        default=False,
    )
    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument('--threshold', type=float, default=0.8)

    return parser

def xyzrgb_array_to_pointcloud2(points, colors, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    assert(points.shape == colors.shape)

    buf = []

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        N = len(points)
        xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
        msg.height = 1
        msg.width = N

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * N
    msg.is_dense = True;
    msg.data = xyzrgb.tostring()

    return msg


def centeroidnp(arr):
    length = arr.shape[0]
    dims = arr.shape[1]
    centroid = np.zeros(dims)

    for i in range(0 , dims):
        centroid[i] = np.sum(arr[:,i])/length
    return centroid

def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y,x,:]
    
    return pc_colors


def backproject(depth_cv, intrinsic_matrix, extrinsic_matrix, return_finite_depth=True, return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)

    #Transform 3D points with the extrinsic matrix
    X = np.vstack(( X, np.ones(X.shape[1]) ))

    extrinsic_matrix = np.vstack(( extrinsic_matrix, np.array([0, 0, 0, 1]) ))
    ext_inv = np.linalg.inv(extrinsic_matrix)
    print (ext_inv.shape)

    X = np.dot(ext_inv,X)

    X = np.vstack(( np.divide(X[0,:],X[3,:]), np.divide(X[1,:],X[3,:]) , np.divide(X[2,:],X[3,:]) ))

    X = np.array(X).transpose()


    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection
        
    return X

## The model is loaded as a global object ##

parser = make_parser()
args = parser.parse_args(sys.argv[1:])
cfg = grasp_estimator.joint_config(
    args.vae_checkpoint_folder,
    args.evaluator_checkpoint_folder,
)
cfg['threshold'] = args.threshold
cfg['sample_based_improvement'] = 1 - int(args.gradient_based_refinement)
cfg['num_refine_steps'] = 10 if args.gradient_based_refinement else 20
estimator = grasp_estimator.GraspEstimator(cfg)
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
config = tf.ConfigProto()
#config=tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf.Session()
estimator.build_network()
estimator.load_weights(sess)

pub = rospy.Publisher('/grasp_proposals', PoseArray )
pub2 = rospy.Publisher("/grasp_points", PointCloud2, queue_size=1000000)
pub3 = rospy.Publisher('/pc_centroid', Marker)

def main():

    rospy.init_node('grasp_detector', anonymous=True)






    '''
    for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')):
        print(npy_file)
        # Depending on your numpy version you may need to change allow_pickle
        # from True to False.
        data = np.load('/home/ahmad3/6dof-graspnet/demo/data/cylinder.npy, allow_pickle=True).item()
    '''



    while not rospy.is_shutdown():

        s = rospy.Service('compute_grasps', computeGrasps, compute_grasps_handle)
        rospy.loginfo('compute_grasps service initialized')
        s.spin()


        '''
        ## Loading data from files ##

        ## Only used when ROS-service is not required ##

        print(data.keys())
        depth = data['depth']
        image = data['image']
        K = data['intrinsics_matrix']


	depth = np.load('/home/ahmad3/6dof-graspnet/demo/realsense_data/realsense_depthImg.npy')
	image = np.load('/home/ahmad3/6dof-graspnet/demo/realsense_data/realsense_rgbImg.npy')
        K = np.load('/home/ahmad3/6dof-graspnet/demo/realsense_data/realsense_K.npy')
        R_ext = np.load('/home/ahmad3/6dof-graspnet/demo/realsense_data/realsense_ext.npy')
        '''

## The inference is run within the service handle ##

def compute_grasps_handle(req):
    print("Request for Grasp compute recieved by graspnet!!")
    #return AddTwoIntsResponse(req.a + req.b)

    ## Subscribe to get images and info from camera ##
    camera_info_topic = '/panda/camera/depth/camera_info'
    depth_img_topic = '/panda/camera/depth/image_raw'
    rgb_img_topic ='/panda/camera/color/image_raw'

    camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    print('Subscribed to Camera Info...')
    depth_image = rospy.wait_for_message(depth_img_topic, Image)
    print('Subscribed to Depth Image...')
    rgb_image = rospy.wait_for_message(rgb_img_topic, Image)
    print('Subscribed to RGB Image...')


    intrinsics = np.asarray(camera_info.K, dtype=np.float64).reshape(3,3)
    #intrinsics = np.vstack((intrinsics, np.array([0,0,0,1])))

    print('Listening to World -> Camera transform...')



    ####  Extract Extrinsics ####
    listener = tf_ros.TransformListener()


    target_frame = 'panda_camera_optical_frame'
    source_frame = 'world'
    while 1:
      try:
        #listener.waitForTransform(target_frame, source_frame, rospy.Time() , rospy.Duration(.1))
        (trans,rot) = listener.lookupTransform(target_frame, source_frame, rospy.Time())
        break
      except (tf_ros.LookupException, tf_ros.ConnectivityException, tf_ros.ExtrapolationException):
        continue
    rot = R.from_quat([rot]).as_dcm()         #conversion from quat to R matrix
    trans = np.transpose(np.array([[trans[0],trans[1],trans[2]]]))

    extrinsics = np.hstack((rot[0], trans))

    depthimgNP = ros_numpy.numpify(depth_image)
    depthimgNP = np.asarray(depthimgNP, dtype=np.float32)
    #depthimgNP = depthimgNP/1000      #Conversion from mm to m -- For ReaslSense Cameras
    rgbimgNP   = ros_numpy.numpify(rgb_image)

    ## TODO: Class implementation of camera2numpy ##
    depth = depthimgNP
    image = rgbimgNP
    K = intrinsics
    R_ext = extrinsics


    ## Removing points that are farther than 1 meter or missing depth  values. ##

    depth[depth == 0] = np.nan
    depth[depth > 1] = np.nan

    ## Backproject to get 3D points ##
    pc, selection = backproject(depth, K, R_ext, return_finite_depth=True, return_selection=True)
    select_point_above_table = 0.4 #0.010 should be changed according to table and camera setup
    pc_ = pc[np.where(pc[:, 2] > select_point_above_table)[0]]

    ## Copy the pointcloud colors from the rgbImg ##
    pc_colors = image.copy()
    pc_colors = np.reshape(pc_colors, [-1, 3])
    pc_colors = pc_colors[selection, :]
    pc_colors = pc_colors[np.where(pc[:, 2] > select_point_above_table)[0]]

    ## output data info ##
    print('Total number of points: %s', pc.shape)
    print('HighestPoint %s', np.nanmax(pc[:,2]))
    print('LowestPoint %s', np.nanmin(pc[:,2]))
    print('Farthest Point %s', np.nanmax(pc[:,0]))
    print('Nearest Point %s', np.nanmin(pc[:,0]))
    print('Total number of points above the table: %s', pc_.shape)

    # Smoothed pc comes from averaging the depth for 10 frames and removing
    # the pixels with jittery depth between those 10 frames.
    #object_pc = data['smoothed_object_pc']
    object_pc = pc_

    points_obj = xyzrgb_array_to_pointcloud2(object_pc, pc_colors, stamp = rospy.Time.now(), frame_id = 'world')
    pub2.publish(points_obj)



    latents = estimator.sample_latents()
    generated_grasps, generated_scores, _ = estimator.predict_grasps(
        sess,
        object_pc,
        latents,
        num_refine_steps=cfg.num_refine_steps,
    )


    generated_scores_np = np.asarray(generated_scores)
    sorting = np.argsort(-generated_scores_np)    #Negative sign is for descending-order sorting
    generated_scores_sorted = generated_scores_np[sorting]

    generated_grasps_np = np.asarray(generated_grasps)
    generated_grasps_sorted = generated_grasps_np[sorting]
    grasp_msgs = []

    obj_centroid = centeroidnp(object_pc)

    #pub3.publish(PointStamped(header=Header(stamp=rospy.Time.now(), frame_id='world'),point= Point(x= obj_centroid[0], y= obj_centroid[1], z= obj_centroid[2])) )
    pub3.publish( Marker(header=Header(stamp=rospy.Time.now(),
                                              frame_id='world'),
                                      pose= Pose(position= Point(x = obj_centroid[0], y = obj_centroid[1],z = obj_centroid[2])),
                                      type=2,
                                      scale=Vector3(0.01,0.01,0.01),
                                      id=0,
                                      color=ColorRGBA(r=1,a=1)))


    ## Publishing grasps as PoseArray for the robot control node ##
    for i in range(0, generated_grasps_sorted.shape[0]):
        this_grasp = Pose()
        this_grasp.position.x = generated_grasps_sorted[i , 0, 3]
        this_grasp.position.y = generated_grasps_sorted[i , 1, 3]
        this_grasp.position.z = generated_grasps_sorted[i , 2, 3]

        r = R.from_dcm([generated_grasps_sorted[i , 0:3, 0:3]])
        r_quat = r.as_quat()[0]


        this_grasp.orientation.x = r_quat[0]
        this_grasp.orientation.y = r_quat[1]
        this_grasp.orientation.z = r_quat[2]
        this_grasp.orientation.w = r_quat[3]

        grasp_msgs.append(this_grasp)

    #rospy.loginfo('Publishing grasps in a PoseArray...')
    #pub.publish(PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = grasp_msgs))    # Publishing not required in the server

    '''
    #Display grasps - commented out of the server
    if len(generated_scores_sorted)!=0:
        draw_scene(
            pc_,
            pc_color=pc_colors_,
            grasps= generated_grasps2,
            grasp_scores= generated_scores2

        )
    else:
        draw_scene(
            pc_,
            pc_color=pc_colors
        )


    mlab.show()

    if len(generated_scores_sorted)!=0:
        draw_scene(
            pc_,
            pc_color=pc_colors_,
            grasps= generated_grasps,
            grasp_scores= generated_scores

        )
    else:
        draw_scene(
            pc_,
            pc_color=pc_colors_
        )
    print('close the window to continue to next object . . .')
    mlab.show()
    '''
    return PoseArray(header=Header(stamp=rospy.Time.now(), frame_id='world'), poses = grasp_msgs)           ## Send grasps in response

if __name__ == '__main__':
    main()


    
