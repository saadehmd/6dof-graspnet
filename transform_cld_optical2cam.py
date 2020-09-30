import rospy
import tf 
import tf2_ros
import tf2_py as tf2
import numpy as np
import ros_numpy
from gazebo_msgs.srv import GetModelState
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointField, PointCloud2
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Quaternion
from std_msgs.msg import Header 


rospy.init_node('tf_cloud_o2c')
pub = rospy.Publisher('/transformed_cloud_camLink', PointCloud2)
#pub2 = rospy.Publisher('/transformed_cloud_world', PointCloud2)


while not rospy.is_shutdown():
	msg = rospy.wait_for_message('panda/camera/depth/points', PointCloud2)
	
	
	ort = tf.transformations.quaternion_from_euler(-1.57, 0, -1.57)
	trans = TransformStamped( header = Header(stamp = rospy.Time.now(), frame_id='panda_camera_link'), transform = Transform( translation = Vector3(x=0,y=0,z=0), rotation = Quaternion(x=ort[0], y=ort[1], z=ort[2], w=ort[3])) )
	cloud_out = do_transform_cloud(msg, trans)
	'''try:
        	rospy.wait_for_service('gazebo/get_model_state')
        	client = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        	resp1 = client('panda_camera', 'world')
	except Exception as inst:
        	print('Error in gazebo/get_link_state service request: ' + str(inst) )


	c1_pos = [resp1.pose.position.x, resp1.pose.position.y, resp1.pose.position.z]
	c1_or = [resp1.pose.orientation.x, resp1.pose.orientation.y, resp1.pose.orientation.z, resp1.pose.orientation.w]
	
	
	trans2 = TransformStamped( header = Header(stamp = rospy.Time.now(), frame_id='world'), transform = Transform( translation = Vector3(x=c1_pos[0],y=c1_pos[1],z=c1_pos[2]), rotation = Quaternion(x=c1_or[0], y=c1_or[1], z=c1_or[2], w=c1_or[3])) )
	cloud_out2 = do_transform_cloud(cloud_out, trans2)'''

	pub.publish(cloud_out)
	#pub2.publish(cloud_out2)
