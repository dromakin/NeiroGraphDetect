# USAGE
# $ python refactor.py ./lines_x_good_JPG ./result

import math
import cv2
import numpy as np
from pip._vendor.distlib.compat import raw_input
from beautifultable import BeautifulTable
from PIL import Image
import imutils
from random import randint
import shutil
import os
import sys

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def scale_image(input_image_path,
                output_image_path,
                width=None,
                height=None):
    table = BeautifulTable()
    table.column_headers = ["INPUT", "SHAPE", "NEW SHAPE"]

    original_image = Image.open(input_image_path)
    w, h = original_image.size

    if answ1 == True:
        print("Resize")

    if width and height:
        max_size = (width, height)
    elif width:
        max_size = (width, h)
    elif height:
        max_size = (w, height)
    else:
        # No width or height specified
        raise RuntimeError('Width or height required!')

    original_image.thumbnail(max_size, Image.ANTIALIAS)

    if answ1 == True:
        x = input_image_path.split("/")
        table.append_row([x[-1], original_image.size, max_size])
        print(table)

    original_image.save(output_image_path)


def resize_image(input_image_path,
                 output_image_path,
                 size):
    table = BeautifulTable()
    table.column_headers = ["INPUT", "SHAPE", "NEW SHAPE"]

    original_image = Image.open(input_image_path)
    width, height = original_image.size

    if answ1 == True:
        print("Resize")

    resized_image = original_image.resize(size)
    width, height = resized_image.size

    if answ1 == True:
        x = input_image_path.split("/")
        table.append_row([x[-1], original_image.size, resized_image.size])
        print(table)

    resized_image.save(output_image_path)


def refactor(angle):
    wastefiles = ['.DS_Store', '.ipynb_checkpoints']
    for img in os.listdir(folderimage):
        if img not in wastefiles:

            table = BeautifulTable()
            table.column_headers = ["INPUT", "SHAPE", "ANGLE", "NEW SHAPE"]

            image = cv2.imread(folderimage + img)
            image_height, image_width = image.shape[0:2]

            ang = 0
            if answ2 == True:
                ang = randint(0, 180)
            else:
                ang = angle

            image_orig = np.copy(image)
            image_rotated = rotate_image(image, ang)
            image_rotated_cropped = crop_around_center(
                image_rotated,
                *rotatedRectWithMaxArea(
                    image_width,
                    image_height,
                    math.radians(ang)
                )
            )

            table.append_row([img, image.shape[0:2], ang, image_rotated_cropped.shape[0:2]])
            if answ1 == True:
                print(table)

            # write images in folder
            # x = img.split(".")
            cv2.imwrite(folder + img, image_rotated_cropped)

            # resize image to input scale
            scale_image(input_image_path=folder + img, output_image_path=resizefolder + img,
                         width=image_width, height=image_height)

            if answ1 == True:
                print('\n\n\n')

# terminal ask
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

                "question" is a string that is presented to the user.
                "default" is the presumed answer if the user just hits <Enter>.
                    It must be "yes" (the default), "no" or None (meaning
                    an answer is required of the user).

                The "answer" return value is True for "yes" or False for "no".
                """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


if __name__ == "__main__":
    # get folder
    folderimage = './' + str(sys.argv[1]) + '/'
    # create folder
    folder = './' + str(sys.argv[2]) + '/'
    createFolder(folder)
    # resize folder
    resizefolder = './' + str(sys.argv[2]) + '_resize/'
    createFolder(resizefolder)

    # ask
    answ1 = query_yes_no('Enable messages in terminal?')
    answ2 = query_yes_no('Enable random angle?')

    angle = 0
    if answ2 == False:
        if answ1 == True:
            print("Write you angle in radians: ")
        angle = float(input())

    refactor(angle)


