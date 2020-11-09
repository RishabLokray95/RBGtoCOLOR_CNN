import os, shutil, time, sys


class PrepareData():
    trainset_dir = "../blue_cis6930/rishab.lokray/images/train/class/"
    testset_dir = "../blue_cis6930/rishab.lokray/images/val/class/"
    userset_dir = "../blue_cis6930/rishab.lokray/images/userInput/class/"

    @classmethod
    def test_train_split(cls):
        os.makedirs(cls.trainset_dir, exist_ok=True)  # 90%
        os.makedirs(cls.testset_dir, exist_ok=True)  # 10%
        os.makedirs(cls.userset_dir, exist_ok=True)

        listings = os.listdir('../blue_cis6930/rishab.lokray/face_images')
        size = len(listings)
        tests_size = size * .1
        for i, file in enumerate(listings):
            if i < tests_size:
                shutil.copyfile('../blue_cis6930/rishab.lokray/face_images/' + file, cls.testset_dir + file)
            else:
                shutil.copyfile('../blue_cis6930/rishab.lokray/face_images/' + file, cls.trainset_dir + file)

