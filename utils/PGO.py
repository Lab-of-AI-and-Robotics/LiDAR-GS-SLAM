import numpy as np
np.set_printoptions(precision=4)

# import minisam
import gtsam
import time
import math

def getConstDigitsNumber(val, num_digits):
    return "{:.{}f}".format(val, num_digits)

def getUnixTime():
    return int(time.time())

def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                     
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def yawdeg2so3(yaw_deg):
    yaw_rad = np.deg2rad(yaw_deg)
    return eulerAnglesToRotationMatrix([0, 0, yaw_rad])

def yawdeg2se3(yaw_deg):
    se3 = np.eye(4)
    se3[:3, :3] = yawdeg2so3(yaw_deg)
    return se3 


def getGraphNodePose(graph, idx):

    pose_out = np.eye(4,4)
    pose = graph.atPose3(gtsam.symbol('x', idx))
    # pose = graph.atPose3(idx)
    pose_out[:3,3] = np.array([pose.x(), pose.y(), pose.z()])
    pose_out[:3,:3] = pose.rotation().matrix()

    return pose_out

class PoseGraphManager:
    def __init__(self, loop_noise):

        # self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
        self.prior_cov = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]))
        # self.const_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        self.const_cov = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        self.const_cov_loop = np.array([loop_noise, loop_noise, loop_noise, loop_noise, loop_noise, loop_noise])
        self.odom_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        # self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov)
        self.loop_cov = gtsam.noiseModel.Diagonal.Sigmas(self.const_cov_loop)

        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_values = gtsam.Values()

        self.opt_param = gtsam.LevenbergMarquardtParams()
        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_values, self.opt_param)

        self.curr_se3 = None
        self.curr_node_idx = None
        self.prev_node_idx = None

        self.graph_optimized = None

    def addPriorFactor(self, idx, pose):

        self.curr_se3 = np.eye(4)

        self.graph_values.insert(gtsam.symbol('x', idx), gtsam.Pose3(pose))
        self.graph_factors.add(gtsam.PriorFactorPose3(
                                                gtsam.symbol('x', idx), 
                                                gtsam.Pose3(pose), 
                                                self.prior_cov))

    def addOdometryFactor(self, idx_1, idx_2, relative_T, initial_T):

        self.graph_values.insert(gtsam.symbol('x', idx_2), gtsam.Pose3(initial_T))
        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', idx_1), 
                                                gtsam.symbol('x', idx_2), 
                                                gtsam.Pose3(relative_T), 
                                                self.odom_cov))

    def addOdometryFactor_re(self, idx_1, idx_2, relative_T):

        # self.graph_values.insert(gtsam.symbol('x', idx_2), gtsam.Pose3(initial_T))
        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', idx_1), 
                                                gtsam.symbol('x', idx_2), 
                                                gtsam.Pose3(relative_T), 
                                                self.odom_cov))


    def addLoopFactor(self, idx_1, idx_2, relative_T):

        self.graph_factors.add(gtsam.BetweenFactorPose3(
                                                gtsam.symbol('x', idx_1), 
                                                gtsam.symbol('x', idx_2), 
                                                gtsam.Pose3(relative_T), 
                                                self.loop_cov))

    def optimizePoseGraph(self):

        self.opt = gtsam.LevenbergMarquardtOptimizer(self.graph_factors, self.graph_values, self.opt_param)
        self.graph_optimized = self.opt.optimize()

        # print(self.graph_optimized)

        # correct current pose
        # pose_trans, pose_rot = getGraphNodePose(self.graph_optimized, 0)
        
    def getValues(self, num_values):
        
        values = []
        
        for i in range(num_values):
            pose = getGraphNodePose(self.graph_optimized, i)
            
            values.append(pose)
        
        return values
        
        