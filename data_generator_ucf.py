import os
import numpy as np
import cv2
import csv
import keras
from skimage import transform
from skimage import util
from keras.utils import np_utils
from random import shuffle


class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, video_dir, depth, train_flag, val_flag):
        self.video_id = 0
        self.batch_size = batch_size
        self.video_dir = video_dir
        self.files = os.listdir(video_dir)
        shuffle(self.files)
        self.depth = depth
        self.train = train_flag
        self.val = val_flag
        self.label_name_number_mapping()
        self.batch_id = 0

    def __len__(self):
        if self.train:
           return int(9573/self.batch_size)
        elif self.val:
            return int(1064/self.batch_size)
        else:
            return int(2719/self.batch_size)

    def label_name_number_mapping(self):
        self.action_mapping = {'ApplyEyeMakeup': 23,
                               'ApplyLipstick': 52,
                               'Archery': 5,
                               'BabyCrawling': 1,
                               'BalanceBeam': 44,
                               'BandMarching': 41,
                               'BaseballPitch': 46,
                               'Basketball': 70,
                               'BasketballDunk': 76,
                               'BenchPress': 53,
                               'Biking': 20,
                               'Billiards': 88,
                               'BlowDryHair': 51,
                               'BlowingCandles': 21,
                               'BodyWeightSquats': 77,
                               'Bowling': 58,
                               'BoxingPunchingBag': 86,
                               'BoxingSpeedBag': 85,
                               'BreastStroke': 68,
                               'BrushingTeeth': 54,
                               'CleanAndJerk': 59,
                               'CliffDiving': 69,
                               'CricketBowling': 3,
                               'CricketShot': 79,
                               'CuttingInKitchen': 71,
                               'Diving': 35,
                               'Drumming': 98,
                               'Fencing': 56,
                               'FieldHockeyPenalty': 99,
                               'FloorGymnastics': 45,
                               'FrisbeeCatch': 6,
                               'FrontCrawl': 13,
                               'GolfSwing': 80,
                               'Haircut': 92,
                               'HammerThrow': 74,
                               'Hammering': 97,
                               'HandStandPushups': 0,
                               'HandstandWalking': 19,
                               'HeadMassage': 4,
                               'HighJump': 22,
                               'HorseRace': 36,
                               'HorseRiding': 64,
                               'HulaHoop': 40,
                               'IceDancing': 78,
                               'JavelinThrow': 38,
                               'JugglingBalls': 27,
                               'JumpRope': 37,
                               'JumpingJack': 24,
                               'Kayaking': 34,
                               'Knitting': 10,
                               'LongJump': 87,
                               'Lunges': 31,
                               'MilitaryParade': 28,
                               'Mixing': 15,
                               'MoppingFloor': 29,
                               'Nunchucks': 82,
                               'ParallelBars': 9,
                               'PizzaTossing': 55,
                               'PlayingCello': 18,
                               'PlayingDaf': 62,
                               'PlayingDhol': 93,
                               'PlayingFlute': 49,
                               'PlayingGuitar': 75,
                               'PlayingPiano': 81,
                               'PlayingSitar': 94,
                               'PlayingTabla': 8,
                               'PlayingViolin': 83,
                               'PoleVault': 16,
                               'PommelHorse': 33,
                               'PullUps': 47,
                               'Punch': 90,
                               'PushUps': 17,
                               'Rafting': 95,
                               'RockClimbingIndoor': 39,
                               'RopeClimbing': 72,
                               'Rowing': 11,
                               'SalsaSpin': 91,
                               'ShavingBeard': 100,
                               'Shotput': 50,
                               'SkateBoarding': 61,
                               'Skiing': 25,
                               'Skijet': 89,
                               'SkyDiving': 43,
                               'SoccerJuggling': 12,
                               'SoccerPenalty': 57,
                               'StillRings': 48,
                               'SumoWrestling': 60,
                               'Surfing': 30,
                               'Swing': 2,
                               'TableTennisShot': 66,
                               'TaiChi': 96,
                               'TennisSwing': 67,
                               'ThrowDiscus': 26,
                               'TrampolineJumping': 7,
                               'Typing': 63,
                               'UnevenBars': 42,
                               'VolleyballSpiking': 84,
                               'WalkingWithDog': 32,
                               'WallPushups': 65,
                               'WritingOnBoard': 73,
                               'YoYo': 14}

    def on_epoch_end(self):
        #print ("\n epoch ended", self.video_loaded_count)
        self.batch_id = 0

    def __getitem__(self, index):
        X,y = self.extract_video_data()
        X = X.transpose((0, 2, 3, 1, 4))
        return X,y

    def extract_video_data(self):

        clips_array = []
        label_array = []
        self.batch_id += 1
        self.clips = 0
        #print("***********************************", self.batch_id)

        while self.clips < self.batch_size:
            try:
                filename = self.files[self.video_id]
            except IndexError as error:
                self.video_id = 0
                filename = self.files[self.video_id]
            #print(filename)
            filepath = os.path.join(self.video_dir, filename)
            cap = cv2.VideoCapture(filepath)

            label = self.get_label_data(filename)
            #print(self.video_id, filename, label)
            label_array.append(np_utils.to_categorical(label, num_classes=101))

            task_clip = self.extract_task_clip(cap)
            clips_array.append(task_clip)

            self.clips +=1
            self.video_id += 1
            cap.release()
        return np.array(clips_array), np.array(label_array)

    def extract_task_clip(self, cap):
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #print("no of frames", nframes)
        frame_bound_ids = np.linspace(0, nframes-1, self.depth+1)
        frame_bound_ids = frame_bound_ids.astype(int)
        required_frame_ids = []
        for i in range(1, self.depth+1):
            required_frame_ids.append(np.random.randint(frame_bound_ids[i-1], frame_bound_ids[i]))
        frames_array = []
        #print(required_frame_ids)
        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, required_frame_ids[i])
            ret, frame = cap.read()
            frame_id = required_frame_ids[i]
            while not ret:
                frame_id = frame_id - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()    
            frame = cv2.resize(frame, (160, 120))
            frames_array.append(frame)
        #print(np.shape(frames_array))
        return frames_array

    def get_label_data(self, filename):
        return self.action_mapping[filename.split('_')[1]]

    def video_preprocessing(self, video):
        rand_int = np.random.randint(4)
        if rand_int == 0:
            return video
        if rand_int == 1:
            return self.random_rotation(video)
        if rand_int == 2:
            return self.random_noise(video)
        if rand_int == 3:
            return self.horizontal_flip(video)

    def random_rotation(self, video):
        random_degree = np.random.uniform(-25,25)
        rotated_clip = []
        for frame in video:
            rotated_clip.append(transform.rotate(frame, random_degree))
        return rotated_clip

    def random_noise(self, video):
        noisy_frames = []
        for frame in video:
            noisy_frames.append(util.random_noise(frame))
        return noisy_frames

    def horizontal_flip(self, video):
        flipped_images = []
        for frame in video:
            flipped_images.append(frame[:,::-1])
        return flipped_images
