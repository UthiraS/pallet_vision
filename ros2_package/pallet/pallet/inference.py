
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
from .segmentationUtil import SingleImageSegInference
from .detectionUtil import SingleImageDetection

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        
        # Create subscription to the image topic
        self.subscription = self.create_subscription(
            Image,
            'rgb_images',
            self.inference_callback,
            10)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        self.get_logger().info('Inference node has started')

        # SET THE SEGMENTATION MODEL PATH HERE!!!!
        # SEG_MDL_PATH = "./final_model_checkpoints/best_segment.pt"

        package_share_dir = get_package_share_directory('pallet')

        SEG_MDL_PATH = os.path.join(package_share_dir, 'models', 'best_segment.pt')
        self.seg = SingleImageSegInference(SEG_MDL_PATH)

        # SET THE DETECTION MODEL PATH HERE!
        DET_MDL_PATH = os.path.join(package_share_dir, 'models', 'best_detect.pt')
        self.det = SingleImageDetection(DET_MDL_PATH)


    def inference_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Perform your post-processing/inference here
            processed_image = self.process_image(cv_image)
            
            # Log some basic information
            self.get_logger().info(f'Processed image with shape: {processed_image.shape}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_image(self, image):
        """
        Add your post-processing/inference code here
        Args:
            image: RGB image as numpy array (H,W,3)
        Returns:
            processed_image: Processed/inferenced image
        """
        # Example processing (replace with your own)
        processed_image = image.copy()
    
        self.seg.run_inference(processed_image)
        self.det.run_inference(processed_image)

        return processed_image

def main(args=None):
    rclpy.init(args=args)
    inference_node = InferenceNode()
    
    try:
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()