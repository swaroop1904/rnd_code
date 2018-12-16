        # path = '/home/swaroop/ros/rnd_code/src/action_recognition/resources/images'
        # for _ in range(20):
        #     start = time.time()
        #     files = os.listdir(path)
        #     image_files = [val for ind, val in enumerate(files) if re.findall(".png", files[ind])]
        #     for count, image in enumerate(image_files):
        #         data = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR )
        #         data = cv2.resize(data, (406, 306))
        #         self.frames[:,:,count,:] = data
        #         #print "loaded"
        #     clips = self.frames[np.newaxis]
        #     #print "predict start"
        #     action_class = self.model.predict_classes(clips)
        #     #print action_class, "predict end"
        #     print time.time()-start












        # array_1 = np.array([0,1,2,3,4,5,6,7,-1])
        # array_2 = np.array([2,1,0,3,5,6,4,7,-1])
        # array_3 = np.array([4,5,0,1,2,3,6,7,-1])
        # array_4 = np.array([4,0,1,2,3,6,5,7,-1])
        # array_5 = np.array([7,6,3,0,1,2,4,5,-1])
        # if is_learning:
        #     for val in array_1:
        #         self.detected_action_pub.publish(val)
        #     for val in array_2:
        #         self.detected_action_pub.publish(val)
        #     for val in array_3:
        #         self.detected_action_pub.publish(val)
        #     for val in array_4:
        #         self.detected_action_pub.publish(val)
        # else:
        #     for val in array_5:
        #         self.detected_action_pub.publish(val)
