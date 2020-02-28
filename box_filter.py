# -*- coding:UTF-8 -*-
"""
This file used to filter bounding boxes
the bounding boxes which does not contain text will be filtered
"""
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import Tuple, List
from os import path
import xmltodict
import cv2 as cv
import hog_desc
import pickle


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    load dataset for training and test
    Outputs:
        the outputs contains trainning and testing dataset and their labels
    """
    info_file = path.join("..", "box-filter-dataset", "dataset-info.xml")
    with open(info_file, "r") as f:
        content = f.read()
    info_dict = xmltodict.parse(content, encoding="UTF-8")
    # train and test x and labels
    train_x: List[float] = []
    train_label: List[int] = []
    test_x: List[float] = []
    test_label: List[int] = []

    i = 0
    train_size = len(info_dict["root"]["images_train"]["item"])
    for ele in info_dict["root"]["images_train"]["item"]:
        i += 1
        if i % 10 == 0:
            print(f"solved {i} images, total {train_size}")
        path_str = ele["path"]
        label = ele["label"]
        img = cv.imread(path.join("..", "box-filter-dataset", path_str))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (300, 500))
        img = np.array(img, dtype="float")
        img = hog_desc.normalize_img(img)
        img = hog_desc.gamma_calibrate(img)
        img = hog_desc.normalize_gamma_recover(img)
        # descriptor
        img_desc = hog_desc.gray_img_desc(img)
        img_desc = np.array(img_desc, dtype="float")
        img_desc = img_desc.flatten()

        train_x.append(img_desc)
        train_label.append(1 if label == "True" else 0)

    i = 0
    test_size = len(info_dict["root"]["images_test"]["item"])
    for ele in info_dict["root"]["images_test"]["item"]:
        i += 1
        if i % 10 == 0:
            print(f"solved {i} images, total {test_size}")
        path_str = ele["path"]
        label = ele["label"]
        img = cv.imread(path.join("..", "box-filter-dataset", path_str))
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (300, 500))
        img = np.array(img, dtype="float")
        img = hog_desc.normalize_img(img)
        img = hog_desc.gamma_calibrate(img)
        img = hog_desc.normalize_gamma_recover(img)
        # descriptor
        img_desc = hog_desc.gray_img_desc(img)
        img_desc = np.array(img_desc, dtype="float")
        img_desc = img_desc.flatten()

        test_x.append(img_desc)
        test_label.append(1 if label == "True" else 0)

    return np.array(train_x, dtype="float"), np.array(train_label, dtype="int"), \
        np.array(test_x, dtype="float"), np.array(test_label, dtype="int")


def train_random_forest(train_x: np.ndarray, train_label: np.ndarray) -> RandomForestClassifier:
    """
    training random forest classifier
    """
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, n_jobs=-1, random_state=0, verbose=1)
    clf.fit(train_x, train_label)
    return clf


def prediction(clf: RandomForestClassifier, test_x: np.ndarray, test_label: np.ndarray) -> float:
    """
    Prediction on test dataset
    """
    test_label_pred = clf.predict(test_x)
    accu = 0
    test_size = len(test_label)
    for i in range(test_size):
        if test_label_pred[i] == test_label[i]:
            accu += 1
    accu = accu*1.0/test_size
    return accu


if __name__ == "__main__":
    with open("pre_training_data.pkl", "rb") as f:
        train_x, train_label, test_x, test_label = pickle.load(f)
    clf = train_random_forest(train_x, train_label)
    accu = prediction(clf, test_x, test_label)
    print(f"test accuracy = {accu}")
    # save the training model
    with open("random_forest.model", "wb") as f:
        pickle.dump(clf, f)
