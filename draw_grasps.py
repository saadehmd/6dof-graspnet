from visualization_utils import *
import mayavi.mlab as mlab
import pcl
import numpy as np
from scipy.spatial.transform import Rotation as R

a = pcl.load('/home/ahmad3/PVN3D/pvn3d/datasets/openDR/openDR_dataset/models/obj_7.pcd')
b = np.array(a)
c = 200*np.ones((b.shape[0], 3), dtype=np.uint8)
preDefined_grasps = np.loadtxt('/home/ahmad3/PVN3D/pvn3d//datasets/openDR/dataset_config/preDefined_grasps/7.txt')

grasp_lst = []
grasp_scores = []

for i in range(0, preDefined_grasps.shape[0]):
	trans_mat = R.from_euler('zyx', preDefined_grasps[i, 3:6]).as_dcm()
	grasp_pose = np.hstack((trans_mat, preDefined_grasps[i, 0:3].reshape(3,1)))
	grasp_lst.append(grasp_pose)
	grasp_scores.append(100)



draw_scene(b, pc_color=c, grasps= grasp_lst, grasp_scores= grasp_scores)
mlab.show()
