#!/home/ahmad3/anaconda2/envs/mrcnn/bin/python
# -*- coding: utf-8 -*-
import rospy
#import pcl
from sensor_msgs.msg import PointCloud2
#import sensor_msgs.point_cloud2 as pc2
import ros_numpy
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




def main(args):

    parser = make_parser()
    args = parser.parse_args(args)
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





    msg = rospy.wait_for_message('/cloud_pcd', PointCloud2)
    pc = ros_numpy.numpify(msg)
    points=np.zeros((pc.shape[0],3))
    colors=100*np.ones((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    
    latents = estimator.sample_latents()
    generated_grasps, generated_scores, _ = estimator.predict_grasps(sess,points,latents,num_refine_steps=cfg.num_refine_steps,)
    #print(np.array(generated_grasps).shape)
    print(type(generated_grasps))
    scores_np = np.asarray(generated_scores)
    sorting = np.argsort(-scores_np)
    sorted_scores = scores_np[sorting]

    grasps_np = np.asarray(generated_grasps)
    sorted_grasps = grasps_np[sorting]




    mlab.figure(bgcolor=(1,1,1))
    if len(generated_scores)!=0:
        draw_scene(
            points,
            pc_color=colors,
            grasps= sorted_grasps[0:20].tolist(),
            grasp_scores= sorted_scores[0:20].tolist()

        )
    else:
        draw_scene(
            points,
            pc_color=colors,
        )
    print('close the window to continue to next object . . .')
    mlab.show()



if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)
    while not rospy.is_shutdown():
        main(sys.argv[1:])


