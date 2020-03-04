
import os


import rosbag
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
import cv2

if __name__ == '__main__':
    BAGS_DIRECTORY = os.path.join(os.environ['HOME'], 'data/NN_data/robofest_data/__RF19/bags_good/')
    bags_fpaths = [os.path.join(BAGS_DIRECTORY, fname) for fname in os.listdir(BAGS_DIRECTORY) if fname.endswith('.bag')]

    # bags_fpaths = ['/home/alexey/data/NN_data/robofest_data/__RF19/bags_good/cameras_2019-03-20-18-14-03.bag']

    TOPIC_NAMES = ['/signs_camera/image_raw/compressed', '/signs_camera/image_raw']

    OUTPUT_DIRECTORY = os.path.join(BAGS_DIRECTORY, '_frames')

    bridge = CvBridge()

    for bag_fpath in bags_fpaths:

        bag_data = rosbag.Bag(bag_fpath)

        print('reading: {}'.format(bag_fpath))

        BAG_FSTEM = os.path.basename(bag_fpath).split('.')[0]
        BAG_FRAMES_DIR = os.path.join(OUTPUT_DIRECTORY, BAG_FSTEM)
        try:
            os.makedirs(BAG_FRAMES_DIR)
        except Exception as e:
            print(e)

        index = 0

        for topic, msg, t in bag_data.read_messages(topics=TOPIC_NAMES):
            if index % 10 == 0:
                if topic.endswith('compressed'):
                    np_arr = np.fromstring(msg.data, np.uint8)
                    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    result_fname = BAG_FSTEM + '_compressed_{}.png'.format(index)
                    result_fpath = os.path.join(BAG_FRAMES_DIR, result_fname)

                    cv2.imwrite(result_fpath, cv_image)
                else:
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

                        result_fname = BAG_FSTEM + '_{}.png'.format(index)
                        result_fpath = os.path.join(BAG_FRAMES_DIR, result_fname)

                        cv2.imwrite(result_fpath, cv_image)
                    except CvBridgeError as e:
                        print(e)

            index += 1




