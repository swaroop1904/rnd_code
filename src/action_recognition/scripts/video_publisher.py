#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoPublisher():
    def __init__(self, video_path, image_topic, height, width):
        self.image_publisher = rospy.Publisher(image_topic, Image, queue_size=10)
        self.height = height
        self.width = width
        self.video_path = video_path
        self.bridge = CvBridge()

    def read_video(self):
        cap = cv2.VideoCapture(self.video_path)
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in range(0, int(nframes), 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, img = cap.read()
            img = np.array(img)
            ros_img = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.image_publisher.publish(ros_img)

if __name__ == '__main__':
    rospy.init_node('video_publisher_node')
    video_path = rospy.get_param('~video_path', '')
    height = rospy.get_param('~height', 406)
    width = rospy.get_param('~width', 306)
    image_topic = rospy.get_param('~image_topic', 'image_topic')
    image_generator = VideoPublisher(video_path, image_topic, height, width)
    image_generator.read_video()
