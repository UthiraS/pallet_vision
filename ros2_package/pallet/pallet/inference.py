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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')
        
        # Define QoS profile to match the bag
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscription with the correct QoS
        self.subscription = self.create_subscription(
            Image,
            '/robot1/zed2i/left/image_rect_color',
            self.inference_callback,
            qos)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        self.get_logger().info('Inference node has started')

        # Get package share directory
        package_share_dir = get_package_share_directory('pallet')

        # Initialize segmentation model
        SEG_MDL_PATH = os.path.join(package_share_dir, 'models', 'best_segment.pt')
        self.get_logger().info(f'Loading segmentation model from: {SEG_MDL_PATH}')
        self.seg = SingleImageSegInference(SEG_MDL_PATH)

        # Initialize detection model
        DET_MDL_PATH = os.path.join(package_share_dir, 'models', 'best_detect.pt')
        self.get_logger().info(f'Loading detection model from: {DET_MDL_PATH}')
        self.det = SingleImageDetection(DET_MDL_PATH)

        # Initialize frame counter
        self.frame_count = 0
        
        # Set target size for inference
        self.target_width = 640
        self.target_height = 480

    def resize_image(self, image):
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        aspect = width / height
        
        if aspect > (self.target_width / self.target_height):
            new_width = self.target_width
            new_height = int(new_width / aspect)
        else:
            new_height = self.target_height
            new_width = int(new_height * aspect)
            
        return cv2.resize(image, (new_width, new_height))

    def inference_callback(self, msg):
        """Callback for processing incoming images"""
        try:
            # Convert ROS Image message to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert BGR to RGB for the models
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            resized_image = self.resize_image(rgb_image)
            
            # Update frame counter
            self.frame_count += 1
            
            # Process the image
            self.process_image(resized_image)
            
            # Display original image
            cv2.imshow('Original Image', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
            
            # Log progress
            self.get_logger().info(
                f'Processed frame {self.frame_count} - '
                f'Original shape: {rgb_image.shape}, '
                f'Resized shape: {resized_image.shape}'
            )
            
            # Allow visualization windows to update
            cv2.waitKey(3)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            import traceback
            traceback.print_exc()

    def process_image(self, image):
        """
        Process image through both models
        Args:
            image: RGB image as numpy array (H,W,3)
        """
        try:
            # Run segmentation
            self.get_logger().info("Running segmentation...")
            self.seg.run_inference(image.copy())
            
            # Run detection
            self.get_logger().info("Running detection...")
            self.det.run_inference(image.copy())
            
        except Exception as e:
            self.get_logger().error(f'Error in inference: {str(e)}')
            import traceback
            traceback.print_exc()

def main(args=None):
    rclpy.init(args=args)
    inference_node = InferenceNode()
    
    try:
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()