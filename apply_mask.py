import numpy as np
import cv2

from main import apply_Haar_filter, apply_sprite, apply_sprite2feature

haar_faces = cv2.CascadeClassifier("./filters/haarcascade_frontalface_default.xml")
haar_eyes = cv2.CascadeClassifier("./filters/haarcascade_eye.xml")
haar_mouth = cv2.CascadeClassifier("./filters/Mouth.xml")
haar_nose = cv2.CascadeClassifier("./filters/Nose.xml")


def apply_mask(image: np.array, mask_name: str):
    faces = apply_Haar_filter(image, haar_faces, 1.3, 5, 30)

    for (x, y, w, h) in faces:  # if there are faces
        # hat condition
        if mask_name == "glasses_and_mustache":
            apply_sprite2feature(
                image,
                "./sprites/glasses.png",
                haar_eyes,
                0,
                h / 3,
                0,
                False,
                w * 0.9,
                x,
                y,
                w,
                h,
                is_eyes=True
            )
            # empirically mouth is at 2/3 of the face from the top
            # empirically the width of mustache is have of face's width (offset of w/4)
            # we look for mouths only from the half of the face (to avoid false positives)
            apply_sprite2feature(
                image,
                "./sprites/mustache.png",
                haar_mouth,
                w / 4,
                2 * h / 3,
                h / 2,
                True,
                w / 2,
                x,
                y,
                w,
                h,
            )

        elif mask_name == "anime":
            # empirically eyes are at 1/3 of the face from the top
            apply_sprite2feature(
                image,
                "./sprites/more_anime.png",
                haar_eyes,
                0,
                h / 3,
                0,
                False,
                w * 0.9,
                x,
                y,
                w,
                h,
                is_eyes=True
            )

        elif mask_name == "shrek":
            apply_sprite(image, "./sprites/shrek.png", w, x, y)

    return image


if __name__ == "__main__":
    image = cv2.imread("./imgs/sample_face.jpeg")

    processed = apply_mask(image, "glasses_and_mustache")
    cv2.imshow("img", image)
    cv2.waitKey(0)
