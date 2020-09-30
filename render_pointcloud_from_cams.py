#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointField, PointCloud2
import tf2_ros
import tf2_py as tf2
import numpy as np
import ros_numpy
import IPython
import pcl

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
    
rospy.init_node('render_pcl')
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
pub = rospy.Publisher('/rendered_cloud', PointCloud2)

while not rospy.is_shutdown():

    cam1_msg = rospy.wait_for_message('/kinect1/depth/points', PointCloud2)
    cam2_msg = rospy.wait_for_message('/kinect2/depth/points', PointCloud2)
    cam3_msg = rospy.wait_for_message('/kinect3/depth/points', PointCloud2)
    print('Listening to transforms...')
    try:
        trans1 = tf_buffer.lookup_transform('world', cam1_msg.header.frame_id,
                                               rospy.Time(0),
                                               rospy.Duration(10))
        trans2 = tf_buffer.lookup_transform('world', cam2_msg.header.frame_id,
                                              rospy.Time(0),
                                              rospy.Duration(10))

        trans3 = tf_buffer.lookup_transform('world', cam3_msg.header.frame_id,
                                             rospy.Time(0),
                                             rospy.Duration(10))
    except tf2.LookupException as ex:
        rospy.logwarn(ex)
        pass
    except tf2.ExtrapolationException as ex:
        rospy.logwarn(ex)
        pass

    cloud_out1 = do_transform_cloud(cam1_msg, trans1)
    cloud_out2 = do_transform_cloud(cam2_msg, trans2)
    cloud_out3 = do_transform_cloud(cam3_msg, trans3)

    cloud_np1 = ros_numpy.numpify(cloud_out1)
    cloud_np2 = ros_numpy.numpify(cloud_out2)
    cloud_np3 = ros_numpy.numpify(cloud_out3)

    print('Clouds Transformed to fixed frame.')
    points1=np.zeros((cloud_np1.shape[0],3))
    points2=np.zeros((cloud_np2.shape[0],3))
    points3=np.zeros((cloud_np3.shape[0],3))

    #colors=100*np.ones((pc.shape[0],3))
    points1[:,0]=cloud_np1['x']
    points1[:,1]=cloud_np1['y']
    points1[:,2]=cloud_np1['z']
    points2[:,0]=cloud_np2['x']
    points2[:,1]=cloud_np2['y']
    points2[:,2]=cloud_np2['z']
    points3[:,0]=cloud_np3['x']
    points3[:,1]=cloud_np3['y']
    points3[:,2]=cloud_np3['z']

    merged_points = np.vstack((points1,points2,points3))
    ## Filtering out points of camera-body in other camera-views
    euc_dist = np.linalg.norm(merged_points, axis=1)
    merged_points = merged_points[np.where(euc_dist < 0.5)]	## We know from the setup, that all cameras are 0.5 m(euc distance) away from the center of rendered view
    print('Clouds merged Now publishing')
    colors = 55*np.ones((merged_points.shape[0],3))
    merged_pc2 = xyzrgb_array_to_pointcloud2(merged_points, colors, stamp=None, frame_id='world', seq=None)
    #IPython.embed()
    pub.publish(merged_pc2)

    merged_cloud = pcl.PointCloud()
    merged_points = np.array(merged_points, dtype=np.float32)
    merged_cloud.from_array(merged_points)
    print('Writing '+ str(merged_points.shape[0])+' points to pcd file')
    pcl.save(merged_cloud, '/home/ahmad3/Desktop/opDR_cadFiles/newFile.pcd')

