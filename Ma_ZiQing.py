import numpy as np
from matplotlib import pyplot as plt
from skimage import io, util, img_as_float, img_as_ubyte, color, feature, measure
from skimage.transform import warp
from skimage.transform import SimilarityTransform, ProjectiveTransform
from sklearn.cluster import KMeans
from scipy import spatial
import os

os.chdir(os.path.dirname(__file__))


def part1():
    # Read input image as grayscale
    img = io.imread("PeppersBayerGray.bmp", as_gray=True)
    # Read input image
    og = io.imread("PeppersRGB.bmp")
    # Get row and col of image
    h, w = img.shape

    """
    The three channels masks are based on this 4 x 4 "sliding window"
    where each letter/location can be matched according to their 
    corresponding demosaicing rules. Locations where the mask is 1 is
    directly copied from the Bayer image and where else is 0 is the
    average of existing neighboring pixels. The number of neighboring
    pixels to average may be 2 (above/below or right/left)
    or 4 (all diagonal corners). If a row/column in the mask is completely
    empty, then values are copied from neighboring non-empty row/column
             Green    Red      Blue
    A B C D  1 0 1 0  0 1 0 1  0 0 0 0
    E F J H  0 1 0 1  0 0 0 0  1 0 1 0
    I J K L  1 0 1 0  0 1 0 1  0 0 0 0
    M N O P  0 1 0 1  0 0 0 0  1 0 1 0
    """
    # reconstruction of the green channel IG
    ig = np.copy(img)  # copy the image into green channel

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            ig[row, col + 1] = (int(img[row, col]) + int(img[row, col + 2])) / 2  # B
            ig[row, col + 3] = (int(img[row, col + 2]) + int(img[row + 1, col + 3])) / 2  # D

            ig[row + 1, col] = (int(img[row, col]) + int(img[row + 2, col])) / 2  # E
            ig[row + 1, col + 2] = (int(img[row + 1, col + 1]) + int(img[row + 1, col + 3]) + int(
                img[row, col + 2]) + int(img[row + 2, col + 2])) / 4  # G

            ig[row + 2, col + 1] = (int(img[row + 2, col]) + int(img[row + 2, col + 2]) + int(
                img[row + 1, col + 1]) + int(img[row + 3, col + 1])) / 4  # J
            ig[row + 2, col + 3] = (int(img[row + 1, col + 3]) + int(img[row + 3, col + 3])) / 2  # L

            ig[row + 3, col] = (int(img[row + 2, col]) + int(img[row + 3, col + 1])) / 2  # M
            ig[row + 3, col + 2] = (int(img[row + 3, col + 1]) + int(img[row + 3, col + 3])) / 2  # O

    # reconstruction of the red channel IR
    ir = np.copy(img)

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            ir[row, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3])) / 2  # C
            ir[row + 1, col + 1] = (int(img[row, col + 1]) + int(img[row + 2, col + 1])) / 2  # F
            ir[row + 1, col + 2] = (int(img[row, col + 1]) + int(img[row, col + 3]) + int(img[row + 2, col + 1]) + int(
                img[row + 2, col + 3])) / 4  # G
            ir[row + 1, col + 3] = (int(img[row, col + 3]) + int(img[row + 2, col + 3])) / 2  # H
            ir[row + 2, col + 2] = (int(img[row + 2, col + 1]) + int(img[row + 2, col + 3])) / 2  # K

            # The first column and last row are filled by copying
            # the second column and the second last row, respectively
            ir[:, col] = ir[:, col + 1]
            ir[row + 3, :] = ir[row + 2, :]

    # reconstruction of the blue channel IB
    ib = np.copy(img)

    for row in range(0, h, 4):  # loop step is 4 since our mask size is 4.
        for col in range(0, w, 4):  # loop step is 4 since our mask size is 4.
            ib[row + 1, col + 1] = (int(img[row + 1, col]) + int(img[row + 1, col + 2])) / 2  # F
            ib[row + 2, col] = (int(img[row + 1, col]) + int(img[row + 3, col])) / 2  # I
            ib[row + 2, col + 1] = (int(img[row + 1, col]) + int(img[row + 1, col + 2]) + int(img[row + 3, col]) + int(
                img[row + 3, col + 2])) / 4  # J
            ib[row + 2, col + 2] = (int(img[row + 1, col + 2]) + int(img[row + 3, col + 2])) / 2  # K
            ib[row + 3, col + 1] = (int(img[row + 3, col]) + int(img[row + 3, col + 2])) / 2  # N

            # The last column and first row are filled by copying
            # the second last column and the second row, respectively
            ib[row, :] = ib[row + 1, :]
            ib[:, col + 3] = ib[:, col + 2]

    # merge the channels
    rgb = np.dstack((ir, ig, ib))

    # Displays the difference of the original image and the merged
    diff = util.compare_images(og, rgb)

    # Create figure with 1 x 3 grid of subplots and set figure size to 20 x 20 inches - 2000 x 2000 pixels
    # Images are displayed with their corresponding titles
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    axes[0].imshow(og)
    axes[0].set_title("Original")

    axes[1].imshow(rgb)
    axes[1].set_title("RGB")

    axes[2].imshow(diff)
    axes[2].set_title("Diff")

    # Automatically adjust subplot parameters and display all subplots
    plt.tight_layout()
    plt.show()


# Finds the closest colour in the palette using kd-tree.
def nearest(palette, colour):
    dist, i = palette.query(colour)
    return palette.data[i]


# Make a kd-tree palette from the provided list of colours
def makePalette(colours):
    return spatial.KDTree(colours)


# Dynamically calculates an N-colour palette for the given image
# Uses the KMeans clustering algorithm to determine the best colours
# Returns a kd-tree palette with those colours
def findPalette(image, ncolours):
    # A copy of the image array and transformed to a 2D numpy array
    arr = np.reshape(image, (-1, 3))
    kmeans = KMeans(n_clusters=ncolours).fit(arr)
    colours = kmeans.cluster_centers_
    # Create a kd-tree palette from the colours computed
    return makePalette(colours)


def FloydSteinbergDitherColor(image, palette):
    # ***** The following pseudo-code is grabbed from Wikipedia:
    # https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering.
    #   for each y from top to bottom ==>(height)
    #    for each x from left to right ==> (width)
    #       oldpixel  := pixel[x][y]
    #       newpixel  := nearest(oldpixel) # Determine the new colour for the current pixel
    #       pixel[x][y]  := newpixel
    #       quant_error  := oldpixel - newpixel
    #       pixel[x + 1][y    ] := pixel[x + 1][y    ] + quant_error * 7 / 16
    #       pixel[x - 1][y + 1] := pixel[x - 1][y + 1] + quant_error * 3 / 16
    #       pixel[x    ][y + 1] := pixel[x    ][y + 1] + quant_error * 5 / 16
    #       pixel[x + 1][y + 1] := pixel[x + 1][y + 1] + quant_error * 1 / 16

    pixel = np.copy(image)
    row, col, d = pixel.shape
    for y in range(row - 1):
        for x in range(col - 1):
            oldpixel = image[x, y]
            newpixel = nearest(palette, oldpixel)
            pixel[x, y] = newpixel
            quant_error = oldpixel - newpixel
            pixel[x + 1, y] += quant_error * 7 / 16
            pixel[x - 1, y + 1] += quant_error * 3 / 16
            pixel[x, y + 1] += quant_error * 5 / 16
            pixel[x + 1, y + 1] += quant_error * 1 / 16

    return pixel


def part2():
    # Read input image as RGB
    image = io.imread("lena.png")

    # Strip the alpha channel if it exists
    image = image[:, :, :3]

    # Convert the image from 8bits per channel to floats in each channel for precision
    image = img_as_float(image)

    ncolours = 8  # The number of colours: change to generate a dynamic palette

    # Dynamically generate an N colour palette for the given image
    palette = findPalette(image, ncolours)

    out = FloydSteinbergDitherColor(image, palette)

    # Create figure with 1 x 2 grid of subplots and set figure size to 10 x 10 inches - 1000 x 1000 pixels
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
    # Images to be displayed are converted back to uint8
    axes[0].imshow(img_as_ubyte(image))
    axes[0].set_title("Original")

    axes[1].imshow(img_as_ubyte(out))
    axes[1].set_title("Dithered")

    plt.tight_layout()
    plt.show()


# Calculates the new coordinates after applying the transformation matrix
def tf_coords(x, y, tf):
    coords = np.array([x, y, 1])
    x_out, y_out, _ = tf @ coords
    return x_out, y_out


# Nearest neighbor interpolation implementation based on advice
# provided by TA (Bernal Manzanilla) during meeting
# and Python implementation from
# https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/ - Adam McQuistan
def nn_coords(x, y, img, tf_inv):
    x_max, y_max = img.shape[0] - 1, img.shape[1] - 1
    # Calculates the new coordinates based on the inverted transformation matrix
    x_out, y_out = tf_coords(x, y, tf_inv)

    if np.floor(x_out) is x_out and np.floor(y_out) is y_out:
        return x_out, x_out

    if np.abs(np.floor(x_out) - x) < np.abs(np.ceil(x_out) - x_out):
        x_out = int(np.floor(x_out))
    else:
        x_out = int(np.ceil(x_out))

    if np.abs(np.floor(y_out) - y_out) < np.abs(np.ceil(y_out) - y_out):
        y_out = int(np.floor(y_out))
    else:
        y_out = int(np.ceil(y_out))

    if x_out > x_max:
        x_out = x_max

    if y_out > y_max:
        y_out = y_max

    return x_out, y_out


def part3():
    # Read input image
    img = io.imread("lab5_img.jpeg")
    # Get row and col of image
    h, w, _ = img.shape

    # Set scale to 2 for x and y
    sx, sy = 2, 2
    # Set angle of rotation to 90 degrees clockwise
    a = np.radians(90)
    sin = np.sin(a).astype(int)
    cos = np.cos(a).astype(int)

    tf_scale = np.array([[sx, 0, 0],
                         [0, sy, 0],
                         [0, 0, 1]])

    tf_rotate = np.array([[cos, sin, 0],
                          [-sin, cos, 0],
                          [0, 0, 1]])
    # Combine the two transformation matrices
    tf = tf_scale @ tf_rotate
    # Invert the combined matrix for inverse mapping
    tf_inv = np.linalg.inv(tf)

    # Create zero-filled numpy array based on newly scaled size
    img_transformed = np.zeros((h * sx, w * sy, 3), dtype=np.uint8)
    # Create copy of array to be filled with new interpolated values
    img_interpolated = np.copy(img_transformed)
    # Every pixel in the image is iterated and their corresponding locations
    # are used to calculate the new positions after applying the transformation
    for x, row in enumerate(img):
        for y, col in enumerate(row):
            x_out, y_out = tf_coords(x, y, tf)
            # Pixels that match are copied over from the original image to the transformed array
            img_transformed[x_out, y_out, :] = img[x, y, :]

    # Every pixel in the transformed image is iterated and their corresponding locations
    # are used to calculate the new positions after applying the transformation
    for x, row in enumerate(img_transformed):
        for y, col in enumerate(row):
            # Find the new coordinates using nearest neighbor interpolation
            x_out, y_out = nn_coords(x, y, img, tf_inv)
            img_interpolated[x, y, :] = img[x_out, y_out, :]

    # Create figure with 1 x 3 grid of subplots and set figure size to 15 x 15 inches - 1500 x 1500 pixels
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 15))
    # Display the original image, transformed image, and interpolated image (fixed aliasing)
    axes[0].imshow(img)
    axes[0].set_title("Original")

    axes[1].imshow(img_transformed)
    axes[1].set_title("2X Scaled and Rotated 90° CW")

    axes[2].imshow(img_interpolated)
    axes[2].set_title("Interpolated")

    plt.tight_layout()
    plt.show()


# An alpha channel is added to the warped images before merging them into a single image
def add_alpha(image, background=-1):
    """
    Add an alpha layer to the image.
    The alpha layer is set to 1 for foreground
    and 0 for background.
    """
    rgb = color.gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))


def part4():
    # Read the input images
    image0 = io.imread("im1.jpg", as_gray=True)
    image1 = io.imread("im2.jpg", as_gray=True)

    # Feature detection and matching

    # Initiate ORB detector
    orb = feature.ORB(n_keypoints=5000)

    # Find the keypoints and descriptors
    orb.detect_and_extract(image0)
    keypoints0 = orb.keypoints
    descriptors0 = orb.descriptors

    orb.detect_and_extract(image1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    # initialize Brute-Force matcher and exclude outliers. See match descriptor function.
    matches = feature.match_descriptors(descriptors0, descriptors1, metric="euclidean", cross_check=True)

    # Compute homography matrix using ransac and ProjectiveTransform
    src = keypoints1[matches[:, 1]][:, ::-1]
    dst = keypoints0[matches[:, 0]][:, ::-1]

    model_robust, inliers = measure.ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1)
    # model_robust, inliers = ransac ...

    # Warping next, we produce the panorama itself. The first step is to find the shape of the output image by
    # considering the extents of all warped images.

    r, c = image1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape = np.ceil(output_shape[::-1]).astype(int)

    # The images are now warped according to the estimated transformation model.

    # A shift is added to ensure that both images are visible in their entirety.
    # Note that warp takes the inverse mapping as input.
    offset = SimilarityTransform(translation=-corner_min)
    tform = (model_robust + offset)
    image0_warped = warp(image0, offset.inverse,
                         output_shape=output_shape)

    image1_warped = warp(image1, tform.inverse,
                         output_shape=output_shape)

    # add alpha to the image0 and image1
    image0_alpha = add_alpha(image0_warped)
    image1_alpha = add_alpha(image1_warped)

    # merge the alpha added image
    merged = (image0_alpha + image1_alpha)

    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel.  Divide by the number of images to get
    # an average.
    alpha = merged[..., 3]
    merged /= np.maximum(alpha, 1)[..., np.newaxis]

    # show output image
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
    axes[0].imshow(image0, cmap="gray")
    axes[0].set_title("Image 0")

    axes[1].imshow(image1, cmap="gray")
    axes[1].set_title("Image 1")

    axes[2].imshow(img_as_ubyte(merged), cmap="gray")
    axes[2].set_title("Merged")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    part1()
    part2()
    part3()
    part4()
