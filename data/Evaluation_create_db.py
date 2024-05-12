import argparse
from os import path

import cv2
import numpy as np
from tqdm import tqdm

from TYY_utils import get_meta


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    db = args.db
    img_size = args.img_size
    min_score = args.min_score

    root_path = "./{}_crop/".format(db)
    mat_path = root_path + "{}.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

    out_genders = []
    out_ages = []
    out_imgs = []
    out_file_paths = []

    for i in tqdm(range(len(face_score))):
        if face_score[i] < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        out_genders.append(int(gender[i]))
        out_ages.append(age[i])

        # Modification to use small dataset
        image_path = root_path + str(full_path[i][0])
        if not path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        out_imgs.append(cv2.resize(img, (img_size, img_size)))
        out_file_paths.append(image_path)

    np.savez(output_path, image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages),
             img_size=img_size, file_path=np.array(out_file_paths))


if __name__ == '__main__':
    main()
