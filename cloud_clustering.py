import pcl
import rospy
import numpy as np
import ros_numpy as rnp
from sensor_msgs.msg import PointCloud2, PointField
import IPython
from pcl_helper import *

# Euclidean Clustering

class cloud_clustering:
	
	def __init__(self,cloud_in):
		
		self.cloud = cloud_in
		self.publisher = rospy.Publisher('/cloud_clustered', PointCloud2)
		self.cluster_indices = None
		self.clustered_cloud = None
		self.clusters = []
	def euclid_cluster(self):
		
		## Estimate clusters ##
		white_cloud = XYZRGB_to_XYZ(self.cloud) # Apply function to convert XYZRGB to XYZ     
		tree = white_cloud.make_kdtree()
		ec = white_cloud.make_EuclideanClusterExtraction()
		ec.set_ClusterTolerance(0.01)
		ec.set_MinClusterSize(20)
		ec.set_MaxClusterSize(30000)
		ec.set_SearchMethod(tree)
		cluster_indices = ec.Extract()
		self.cluster_indices = cluster_indices
		
		## Store clusters as a list
		for j, indices in enumerate(self.cluster_indices):
			points = []
			for i, indice in enumerate(indices):
			    points.append([self.cloud[indice][0],self.cloud[indice][1],self.cloud[indice][2]])
			cloud = pcl.PointCloud()
			cloud.from_list(points)
			self.clusters.append(cloud)
		



	def cluster_mask(self):
		# Create Cluster-Mask Point Cloud to visualize each cluster separately
		#Assign a color corresponding to each segmented object in scene
		cluster_color = get_color_list(len(self.cluster_indices))

		color_cluster_point_list = []

		for j, indices in enumerate(self.cluster_indices):
			for i, indice in enumerate(indices):
			    color_cluster_point_list.append([
				                            self.cloud[indice][0],
				                            self.cloud[indice][1],
				                            self.cloud[indice][2],
				                            rgb_to_float( cluster_color[j] )
				                           ])

		#Create new cloud containing all clusters, each with unique color
		self.clustered_cloud = pcl.PointCloud_PointXYZRGB()
		self.clustered_cloud.from_list(color_cluster_point_list)
		


	def cluster_publish(self):
		clustered_pc2 = pcl_to_ros(self.clustered_cloud)
		self.publisher.publish(clustered_pc2)
		
def main():

	
	cloud_msg = rospy.wait_for_message('/output', PointCloud2)
	print('Subscribed to pointcloud. Clustering...')

	clc = cloud_clustering(cloud_msg)
	clc.euclid_cluster()
	clc.cluster_mask()
	clc.cluster_publish()
	#IPython.embed()
	print('No. of clusters in pointcloud: ', len(clc.clusters))
	

	
if __name__=='__main__':

	rospy.init_node('Cluster_publisher')

	
	
	while not rospy.is_shutdown():
		main() 	
	
