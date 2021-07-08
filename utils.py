import os
import glob
import trimesh
import trimesh.sample
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def create_point_cloud_dataset(data_dir, num_points_per_cloud=1024):
    """
    Given the path to the ModelNet10 dataset, samples the models and creates point clouds
    :param data_dir: path to the ModelNet10 dataset
    :type data_dir: str
    :param num_points_per_cloud: number of points to sample per cloud. 1024, 2048....
    :type num_points_per_cloud: int
    :return: tuple of numpy array containing training and test point clouds, their corresponding labels and a list of
    class IDs
    :rtype: tuple
    """

    train_pc = []   # array of training point clouds
    test_pc = []    # array of test point clouds

    train_labels = []   # array of corresponding training labels
    test_labels = []    # array of corresponding test labels

    class_ids = {}   # list of class names

    # get all the folders except the readme file
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for class_id, folder in enumerate(folders):
        class_name = os.path.basename(folder)
        print("processing class: {}".format(class_name))
        print(class_id)
        # TODO: Fill this part, get the name of the folder (class) and save it
        class_ids[class_id] = class_name

        # get the files in the train folder
        train_files = glob.glob(os.path.join(folder, "train/*"))
        for f in train_files:
            # TODO: Fill this part
            train_pc.append(getPointsfromPath(f, num_points_per_cloud))
            train_labels.append(class_id)
        # get the files in the test folder
        test_files = glob.glob(os.path.join(folder, "test/*"))
        for f in test_files:
            # TODO: FIll this part
            test_pc.append(getPointsfromPath(f, num_points_per_cloud))
            test_labels.append(class_id)

    return (np.array(train_pc), np.array(test_pc),
            np.array(train_labels), np.array(test_labels), class_ids)


def visualize_cloud(point_cloud):
    """
    Utility function to visualize a point cloud
    :param point_cloud: input point cloud
    :type point_cloud: numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    plt.show()


def add_noise_and_shuffle(point_cloud, label):
    """
    Adds noise to a point cloud and shuffles it
    :param point_cloud: input point cloud
    :type point_cloud: tensor
    :param label: corresponding label
    :type label: tensor
    :return: the processed point cloud and the label
    :rtype: tensors
    """
    dev_in_metres = 0.001   # <- change this value to change amount of noise
    # add noise to the points
    point_cloud += tf.random.uniform(point_cloud.shape, -dev_in_metres, dev_in_metres, dtype=tf.float64)
    # shuffle points
    point_cloud = tf.random.shuffle(point_cloud)
    return point_cloud, label


def getPointsfromPath(fname, points_per_cloud):
    cad_mesh = trimesh.load(fname)
    points = trimesh.sample.sample_surface(cad_mesh, points_per_cloud)[0]
    return points


def encode_one_hot(index):
    vec = np.zeros(10)
    vec[index] = 1
    return vec

def display_batch(pointclouds, labels, class_dict, preds = None):
    """ assumes a batch size of 16
    """
    fig = plt.figure()
    for idx in range(pointclouds.shape[0]):
        ax = fig.add_subplot(8, 8, idx+1, projection="3d")
        ax.scatter(pointclouds[idx, :, 0], pointclouds[idx, :, 1], pointclouds[idx, :, 2])
        t = "label: {}".format(class_dict[labels[idx].numpy()])
        if preds is not None:
            t += "; pred: {}".format(class_dict[preds[idx].numpy()])
        ax.set_title(t)
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


