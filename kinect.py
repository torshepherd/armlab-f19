import cv2
import numpy as np
from scipy.linalg import block_diag
from PyQt4.QtGui import QImage
import freenect


class Kinect():
    def __init__(self):
        self.currentVideoFrame = np.array([])
        self.currentDepthFrame = np.array([])
        self.currentDepthMeter = np.array([])
        if(freenect.sync_get_depth() == None):
            self.kinectConnected = False
        else:
            self.kinectConnected = True
        
        # mouse clicks & calibration variables
        self.depth2rgb_affine = np.float32([[1,0,0],[0,1,0]])
        self.kinectCalibrated = False
        self.last_click = np.array([0,0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5,2),int)
        self.depth_click_points = np.zeros((5,2),int)
        self.calib_mat, self.dist_coeff = self.loadCameraCalibration('calibration.cfg')
        self.h = 480
        self.w = 640
        self.fixed_coords = np.float32([[304,304], [304,-304], [-304,-304], [-304,304]])
        self.extrinsic = np.float32([[1,0,0],[0,1,0]])
        # x: 60.7; y: 60.8
         
        """ Extra arrays for colormaping the depth image"""
        self.DepthHSV = np.zeros((480,640,3)).astype(np.uint8)
        self.DepthCM=np.array([])

        """ block info """
        self.block_contours = np.array([])


    def captureVideoFrame(self):
        """                      
        Capture frame from Kinect, format is 24bit RGB    
        """
        if(self.kinectConnected):
            self.currentVideoFrame = freenect.sync_get_video()[0]
        else:
            self.loadVideoFrame()
        self.processVideoFrame()
        

    def processVideoFrame(self):
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.calib_mat, 
                                                               self.dist_coeff,
                                                               (self.w, self.h),
                                                               1,
                                                               (self.w, self.h))
        rgb_frame = cv2.cvtColor(self.currentVideoFrame, cv2.COLOR_RGB2BGR)
        undistorted = cv2.undistort(rgb_frame, 
                                    self.calib_mat, 
                                    self.dist_coeff, 
                                    None, 
                                    new_camera_matrix)
        
        cv2.drawContours(undistorted, self.block_contours, -1, (255,0,255), 3)


    def captureDepthFrame(self):
        """                      
        Capture depth frame from Kinect, format is 16bit Grey, 10bit resolution.
        """
        if(self.kinectConnected):
            if(self.kinectCalibrated):
                self.currentDepthFrame = self.registerDepthFrame(freenect.sync_get_depth()[0])
            else:
                self.currentDepthFrame = freenect.sync_get_depth()[0]
        else:
            self.loadDepthFrame()
        self.Depth2Meters()
        
    
    def Depth2Meters(self):
        self.currentDepthMeter =  0.1236 * np.tan(self.currentDepthFrame / 2842.5 + 1.1863)

    
    def loadVideoFrame(self):
        self.currentVideoFrame = cv2.cvtColor(
            cv2.imread("data/ex0_bgr.png",cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)


    def loadDepthFrame(self):
        self.currentDepthFrame = cv2.imread("data/ex0_depth16.png",0)


    def convertFrame(self):
        """ Converts frame to format suitable for Qt  """
        try:
            img = QImage(self.currentVideoFrame,
                             self.currentVideoFrame.shape[1],
                             self.currentVideoFrame.shape[0],
                             QImage.Format_RGB888
                             )
            return img
        except:
            return None


    def convertDepthFrame(self):
        """ Converts frame to a colormaped format suitable for Qt  
            Note: this cycles the spectrum over the lowest 8 bits
        """
        try:

            """ 
            Convert Depth frame to rudimentary colormap
            """
            self.DepthHSV[...,0] = self.currentDepthFrame
            self.DepthHSV[...,1] = 0x9F
            self.DepthHSV[...,2] = 0xFF
            self.DepthCM = cv2.cvtColor(self.DepthHSV,cv2.COLOR_HSV2RGB)
            cv2.drawContours(self.DepthCM,self.block_contours,-1,(0,0,0),3)

            img = QImage(self.DepthCM,
                             self.DepthCM.shape[1],
                             self.DepthCM.shape[0],
                             QImage.Format_RGB888
                             )
            return img
        except:
            return None


    def getAffineTransform_deprecated(self, coord1, coord2):
        """
        Given 2 sets of corresponding coordinates, 
        find the affine matrix transform between them.

        TODO: Rewrite this function to take in an arbitrary number of coordinates and 
        find the transform without using cv2 functions
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        return cv2.getAffineTransform(pts1, pts2)


    def getAffineTransform(self, coord1, coord2):
        """ Least square to get affine tranformation
        Ax = b, A is 2N*6, b is 2N*1
        Generate N*2 coord1 into 2N*6 matrix
        Generate N*2 coord2 into 2N*1 matrix

        -input:
            -coord1: N*2 ndarray
            -coord2: N*2 ndarray
        -output:
            -affine_mt: 2*3 ndarray from coord2 to coord1
        """
        # generate coord1 into A
        mat_A = np.zeros((2*coord1.shape[0], 6))
        coord1 = np.hstack([coord1, np.ones((coord1.shape[0], 1))])
        for i in range(coord1.shape[0]):
            row = coord1[i,:]
            row_block = block_diag(row, row)
            assert(row_block.shape == (2,6))
            mat_A[2*i:2*i+2, :] = row_block
        
        # generate coord2 into b
        vec_b = coord2.reshape(-1,1)

        # solve the least square
        pseudo_inv = np.linalg.inv(np.matmul(mat_A.T, mat_A))
        pseudo_inv = np.matmul(pseudo_inv, mat_A.T)
        affine_mat = np.matmul(pseudo_inv, vec_b)
        assert(affine_mat.shape == (6,1))
        
        return affine_mat.reshape(2,-1)


    def registerDepthFrame(self, frame):
        """
        TODO:
        Using an Affine transformation, transform the depth frame to match the RGB frame
        """
        # Create transformed DepthHSV image placeholder
        frame_affine = np.zeros_like(frame, dtype=frame.dtype)
        
        # Index mapping
        rows, cols = frame.shape
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        coord_flatten = np.vstack([x.reshape(-1),                    # shape (rows*cols, 3)
                                  y.reshape(-1), 
                                  np.ones(rows*cols)])
        coord_map = np.matmul(self.depth2rgb_affine, coord_flatten) # shape (2, row*cols)
        
        # get rid of the outliers in the coord_map
        inlierr = np.logical_and((coord_map[0,:] < cols),
                                 (coord_map[0,:] >= 0))
        inlierc = np.logical_and((coord_map[1,:] < rows),
                                 (coord_map[1,:] >= 0))
        inliers = np.logical_and(inlierr, inlierc)
        coord_inliers = (coord_map.T)[inliers].astype(np.int)
        frame_inliers = frame.reshape(-1)[inliers]
        frame_affine[coord_inliers[:,1], coord_inliers[:,0]] = frame_inliers    
        return frame_affine


    def toWorldCoord(self, x, y, z):
        if not self.kinectCalibrated: raise Exception("How could this function be\
                                                      called? Kinect not calibrated yet")
        camera_xy = self.toCamearaCoord(np.float32([[x, y]]))
        w_x, w_y = np.matmul(self.extrinsic, camera_xy.T)
        # w_z = 0.1236 * np.tan(z / 2842.5 + 1.1863)
        w_z = -0.0009 * z**2 - 0.3393 * z + 712.76
        return (w_x, w_y, w_z)
    

    def toCamearaCoord(self, rgb_clicks):
          
        pixel_coord = np.hstack([rgb_clicks, np.ones((rgb_clicks.shape[0], 1))]).T
        camera_coord = np.matmul(np.linalg.inv(self.calib_mat), 
                                 pixel_coord)
        return camera_coord.T
    

    def Pixel2World(self, u, v):
        # change u, v into [x, y, z]
        # get depth in raw depth figure
        z = self.currentDepthFrame[int(v), int(u)]
        # change into w_x, w_y, w_z
        camera_xy = self.toCamearaCoord(np.float32([[u, v]]))
        w_xy = np.matmul(self.extrinsic, camera_xy.T)
        w_z = -0.0009 * z**2 - 0.3393 * z + 712.76
        coord = np.append(w_xy, w_z)
        return coord


    def loadCameraCalibration(self, filepath):
        """
        TODO:
        Load camera intrinsic matrix from file.
        """
        with open(filepath, 'r') as f:
            calib_mat = np.array([]).reshape(0,3)
            for i, line in enumerate(f):
                line = line.replace('[', '')
                line = line.replace(']', '')
                if i in [0, 4]:
                    continue
                elif i == 5:
                    dist_coeff = np.asarray([float(s) for s in line.split()])
                else:
                    calib_mat = np.vstack([calib_mat, np.asarray([float(s) for s in line.split()])])
        return calib_mat, dist_coeff
        
    
    def blockDetector(self):
        """
        TODO:
        Implement your block detector here.  
        You will need to locate
        blocks in 3D space
        """
        pass


    def detectBlocksInDepthImage(self):
        """
        TODO:
        Implement a blob detector to find blocks
        in the depth image
        """
        pass


def add_ones(mat):
    return np.hstack([mat, np.ones((mat.shape[0], 1))])


if __name__ == "__main__":
    kinect = Kinect()

    # Tests for loading camera calibration
    kinect.loadCameraCalibration('calibration.cfg')
    
    # Tests for affine transformation
    # translation by x = 1
    trans_src = np.array([[0,0],[0,1],[1,1],[1,0]])
    trans_dst = np.array([[1,0],[1,1],[2,1],[2,0]]) + 100
    # pure rotation: pi/2 around origin
    rot_src = np.array([[0,0],[0,1],[1,1],[1,0]])
    rot_dst = np.array([[0,0],[-1,0],[-1,1],[0,1]])
    # Mixation
    mix_src = np.array([[0,0],[0,255],[255,255],[255,0]])
    mix_dst = np.array([[255/2,0],[0,255/2],[255/2,255],[255,255/2]])

    trans_affine = kinect.getAffineTransform(trans_src, trans_dst)
    # print("Translation test")
    # print(trans_affine)
    # print(trans_dst)
    # print(np.matmul(trans_affine, add_ones(trans_src).T))

    rot_affine = kinect.getAffineTransform(rot_src, rot_dst)
    # print("Rotation test")
    # print(rot_affine)
    # print(rot_dst)
    # print(np.matmul(rot_affine, add_ones(rot_src).T))
    
    # mix_affine = kinect.getAffineTransform(mix_src, mix_dst)
    
    kinect.depth2rgb_affine = mix_affine
    lena = cv2.imread("/home/student/Downloads/8-bit-256-x-256-Grayscale-Lena-Image.png",
                      cv2.IMREAD_GRAYSCALE)
    lena = np.asarray(lena)
    affine_lena = kinect.registerDepthFrame(lena)
    cv2.imshow('Lena', affine_lena)
    cv2.waitKey()
    
    
    
    
    
