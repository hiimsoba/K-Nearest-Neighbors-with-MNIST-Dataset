import numpy as np
import struct
import cv2

# test_f0name = "images.idx3-ubyte"
# test_f1name = "labels.idx1-ubyte"
f0name = "t10k-images.idx3-ubyte"
f1name = "t10k-labels.idx1-ubyte"

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

if __name__ == '__main__':
    # training data
    images = read_idx(f0name)
    labels = read_idx(f1name)

    # N for how many training images we should keep... cuz big data set = slow, slow af.
    N = 7500

    # formatted images, as a Nx784 array
    f_img = np.empty([N, images.shape[1] * images.shape[2]])

    for i in range(len(f_img)):
        f_img[i] = images[i].ravel()

    print("\nFinished prepping the training data!")

    # and now load the "testing" data
    # which will be 10000 - N data points
    test_images = images[-(images.shape[0] - N):] # read_idx(test_f0name)
    # and the labels to check if we got a correct result
    test_labels = labels[-(images.shape[0] - N):] # read_idx(test_f1name)

    # formatted images, as a Nx784 array
    f_t_img = np.empty([len(test_images), test_images.shape[1] * test_images.shape[2]])

    for i in range(len(f_t_img)):
        f_t_img[i] = test_images[i].ravel()

    print("Finished prepping the testing data!\n")

    # so at this point i have the "training" data - images and labels
    # now i can load an image and compare it with the training data
    # meaning i calculate the distances, sort them and get the K nearest neighbors
    # and the "guess" / result of the algorithm will be the label that appears the most in those K neighbors

    # so first set some K
    K = 5

    correct = 0
    total = 0

    # and now, for each image in the testing set
    for i in range(len(f_t_img)):
        # get the distances from the current "point" to all the other points IN THE TRAINING SET!
        distances = np.linalg.norm(f_img - f_t_img[i], ord=2, axis=1.)
        # and make like a set of pairs (distance, index)
        set_of_pairs = np.empty([len(distances), 2])
        for j in range(len(set_of_pairs)):
            set_of_pairs[j] = [distances[j], j]
        # sort the distances
        # and get the first K ones
        # here's a little lesson of trickery
        knn = set_of_pairs[set_of_pairs[:,0].argsort()][:K]
        # and get the most frequent label and clasify this image as that label
        frequency = {}
        for k in range(len(knn)):
            if labels[int(knn[k][1])] in frequency.keys():
                frequency[labels[int(knn[k][1])]] += 1
            else:
                frequency[labels[int(knn[k][1])]] = 1
        best = 0
        label = -1
        for k in frequency.keys():
            if frequency[k] > best:
                best = frequency[k]
                label = k
        total += 1
        if label == test_labels[i]:
            correct += 1
            print("Correct : " + str((correct / total) * 100) + "% ( " + str(correct) + " out of " + str(total) + " )")

        # print("Guess : " + str(label) + " ", end="", flush=True)
        # print("Correct : " + str(test_labels[i]))

    # # show each image and print its label in the console
    # for i in range(len(images)):
    #     big = cv2.resize(images[i], (512, 512))
    #     cv2.imshow('image', big)
    #     print(labels[i])
    #     cv2.waitKey(250)
