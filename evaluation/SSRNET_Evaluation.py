import os.path
import timeit

import cv2
import numpy as np
from keras import backend as K
from moviepy.editor import *
from mtcnn.mtcnn import MTCNN

from SSRNET_model import SSR_net
from TYY_utils import load_data_npz


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main():
    K.set_learning_phase(0)  # make sure it's testing mode
    weight_file = "../pre-trained/wiki/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    # for face detection
    detector = MTCNN()

    # create a directory to save images if it doesn't exist
    try:
        os.mkdir('./img')
    except OSError:
        pass

    # load model and weights
    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)

    img_idx = 0
    time_detection = 0
    time_network = 0
    ad = 0.4

    # Load images, ages, etc. from the npz file for evaluation
    images, _, ages, image_size, file_paths = load_data_npz('../data/imdb_db.npz')

    predicted_data = []

    # iterate over images in the data
    for idx, filename in enumerate(file_paths):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_idx = idx
            img_path = filename
            input_img = cv2.imread(os.path.join('../data', img_path))

            img_h, img_w, _ = input_img.shape
            input_img = cv2.resize(input_img, (1024, int(1024 * img_h / img_w)))
            img_h, img_w, _ = input_img.shape

            # detect faces using MTCNN
            start_time = timeit.default_timer()
            detected = detector.detect_faces(input_img)
            elapsed_time = timeit.default_timer() - start_time
            time_detection = time_detection + elapsed_time
            faces = np.empty((len(detected), img_size, img_size, 3))

            for i, d in enumerate(detected):
                print(i)
                print(d['confidence'])
                if d['confidence'] > 0.95:
                    x1, y1, w, h = d['box']
                    x2 = x1 + w
                    y2 = y1 + h
                    xw1 = max(int(x1 - ad * w), 0)
                    yw1 = max(int(y1 - ad * h), 0)
                    xw2 = min(int(x2 + ad * w), img_w - 1)
                    yw2 = min(int(y2 + ad * h), img_h - 1)
                    cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            start_time = timeit.default_timer()
            if len(detected) > 0:
                # predict ages of the detected faces
                results = model.predict(faces)
                predicted_data.append((filename, results))

            # draw results
            for i, d in enumerate(detected):
                if d['confidence'] > 0.95:
                    x1, y1, _, _ = d['box']
                    label = "{}".format(int(results[i]))
                    draw_label(input_img, (x1, y1), label)
            elapsed_time = timeit.default_timer() - start_time
            time_network = time_network + elapsed_time

            # display or save the result
            # cv2.imshow("result", input_img)
            # cv2.waitKey(0)
            try:
                os.makedirs("./img/00")
            except OSError:
                pass

            save_path = os.path.join('./img/00', os.path.basename(img_path))
            cv2.imwrite(save_path, input_img)

            # print('Detection Time: ', time_detection)
            # print('Network Time: ', time_network)
            print(f'Image {img_idx} of {len(images)} done')

    # for filename, age in predicted_data:
    #     print(f'{filename}: {int(age)}')

    # Get the indexes of the images that were predicted
    # This is to get the actual ages of the images
    indexes = []
    for filename, _ in predicted_data:
        for i, path in enumerate(file_paths):
            if filename == path:
                indexes.append(i)
                break
    actual_ages = [ages[i] for i in indexes]

    # Calculate the mean absolute error (check for nan)
    predicted_age_int = []
    for _, age in predicted_data:
        try:
            val = int(age)
        except ValueError:
            val = 0
        except TypeError:
            val = int(age[0])

        predicted_age_int.append(val)

    mae = np.mean(np.abs(np.array(actual_ages) - np.array(predicted_age_int)))
    print('Mean Absolute Error:', mae)

    # Make a graph with the mae for different age ranges
    age_ranges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    mae_ranges = [0] * 10
    count_ranges = [0] * 10
    for i, age in enumerate(actual_ages):
        for j, age_range in enumerate(age_ranges):
            if age < age_range:
                mae_ranges[j] += abs(age - predicted_age_int[i])
                count_ranges[j] += 1
                break
    for i in range(10):
        if count_ranges[i] != 0:
            mae_ranges[i] /= count_ranges[i]
    print('Mean Absolute Error for different age ranges:', mae_ranges)


if __name__ == '__main__':
    main()
