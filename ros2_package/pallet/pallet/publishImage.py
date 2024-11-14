#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from glob import glob
import time

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        # Create publisher
        self.publisher_ = self.create_publisher(Image, 'rgb_images', 10)
        
        # Set the publishing rate to 1 Hz
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # SET YOUR FULL PATH HERE !!!!!
        self.image_dir = os.path.expanduser('~/work_no_backup/rosws/src/data/test/images/')
        
        # Get list of image files
        self.image_files = sorted(glob(os.path.join(self.image_dir, '*.jpg')))  # Add other extensions if needed
        self.current_image_index = 0
        
        # Verify images were found
        if not self.image_files:
            self.get_logger().error('No images found in specified directory!')
            return
            
        self.get_logger().info(f'Found {len(self.image_files)} images')

    def timer_callback(self):
        # Check if we have any images
        if not self.image_files:
            return
            
        # Read the current image
        image_path = self.image_files[self.current_image_index]
        cv_image = cv2.imread(image_path)
        
        if cv_image is None:
            self.get_logger().error(f'Failed to read image: {image_path}')
            return
            
        # Convert from BGR to RGB (OpenCV uses BGR by default)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        try:
            # Convert OpenCV image to ROS message
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')
            
            # Add header timestamp
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_frame'
            
            # Publish the image
            self.publisher_.publish(ros_image)
            self.get_logger().info(f'Published image: {os.path.basename(image_path)}')
            
        except Exception as e:
            self.get_logger().error(f'Error converting/publishing image: {str(e)}')
            return
            
        # Move to next image, loop back to start if at end
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()