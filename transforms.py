
import math

import numpy as np
import cv2


############hw3 functions ##########3

def clean_SP_noise_single(im, radius):
    image = im.copy()
    denoised_image = np.zeros_like(image)
    for i in range(radius, im.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            neighborhood = image[i - radius:i + radius + 1, j - radius:j + radius + 1]
            denoised_image[i, j] = np.median(neighborhood)
    return denoised_image


######### hw2 functions ###############


def find_transform(pointset1, pointset2):
    size = pointset1.shape[0]
    X = np.zeros((size * 2, 8))
    X_tag = np.zeros(size * 2)

    for i in range(0, size):
        X[2 * i][6] = -(pointset1[i][0] * pointset2[i][0])
        X[2 * i][7] = -(pointset1[i][1] * pointset2[i][0])
        X[2 * i + 1][6] = -(pointset1[i][0] * pointset2[i][1])
        X[2 * i + 1][7] = -(pointset1[i][1] * pointset2[i][1])

        X[i * 2][0] = pointset1[i][0]
        X[i * 2][1] = pointset1[i][1]
        X[i * 2 + 1][2] = pointset1[i][0]
        X[i * 2 + 1][3] = pointset1[i][1]
        X[i * 2][4] = 1
        X[i * 2 + 1][5] = 1

        X_tag[i * 2] = pointset2[i][0]
        X_tag[i * 2 + 1] = pointset2[i][1]

    T = np.matmul(np.linalg.pinv(X), X_tag)
    copy_T = T.copy()
    T = np.hstack((T, [1]))
    T = np.reshape(T, (3, 3))

    T[1][1] = copy_T[3]
    T[1][0] = copy_T[2]
    T[0][1] = copy_T[1]
    T[0][0] = copy_T[0]
    T[1][2] = copy_T[5]
    T[0][2] = copy_T[4]
    T[2][1] = copy_T[7]
    T[2][0] = copy_T[6]
    T[2][2] = 1

    return T


def trasnform_image(image, T):
    new_image = np.zeros((image.shape[0], image.shape[1]))

    pixel_mat = np.zeros((3, 1))
    # inv_T = np.linalg.inv(T)
    inv_T = np.linalg.inv(T)

    for i in range(0, new_image.shape[1]):
        for j in range(0, new_image.shape[0]):
            pixel_mat[0][0] = i
            pixel_mat[1][0] = j
            pixel_mat[2][0] = 1
            new_pixel_mat = np.matmul(inv_T, pixel_mat)
            x = round(new_pixel_mat[0][0] / new_pixel_mat[2][0])
            y = round(new_pixel_mat[1][0] / new_pixel_mat[2][0])
            if 0 <= x < new_image.shape[1] and 0 <= y < new_image.shape[0]:
                new_image[j][i] = image[y][x]

            else:
                new_image[j][i] = 0

    # new_image = np.transpose(new_image)
    # # new_image = np.transpose(new_image)
    return new_image


def clean_SP_noise_multiple(images):
    return np.median(images, axis=0).squeeze()


####################### hw1 functions ###########################################


def histImage(im):
    h = [0] * 256

    for i in im:
        for j in i:
            h[j] += 1

    return h


def nhistImage(im):
    hist1 = histImage(im)

    nh = [0] * 256
    sum = 0
    for i in hist1:
        sum += i

    for i in range(0, 256):
        nh[i] = (hist1[i] / sum)

    return nh


def ahistImage(im):
    hist = histImage(im)

    ah = [0] * 256
    for i in range(0, 255):
        sum = 0
        for j in range(0, i):
            sum += hist[j]
        ah[i] = sum

    return ah


def calcHistStat(h):
    m = np.dot(h, np.ones(256) / 256)

    e = (np.dot(np.matmul(h, np.diag(h)), np.ones(256)) / 256) - math.pow(m, 2)

    return m, e


def mapImage(im, tm):
    nim = [[tm[i] for i in row] for row in im]
    nim = [[min(255, max(0, i)) for i in row] for row in nim]
    return nim


def histEqualization(im):
    accum_of_orig = ahistImage(im)

    mean, exp = calcHistStat(histImage(im))

    mean_hist = [mean] * 256

    accum_of_mean = np.cumsum(mean_hist)

    tm = 0 * np.arange(256)
    index_mean = 0
    index_orig = 0

    for index, i in enumerate(accum_of_mean):
        if index_orig > 255:
            break
        while i > accum_of_orig[index_orig]:
            tm[index_orig] = index
            index_orig += 1
            if index_orig > 255:
                break

    return tm





def clean_baby(im):
    im1 = im.copy()

    dst1 = np.array([[0, 0], [0, 255], [255, 255],[255,0]]).astype(np.float32)

    src1 = np.array([[6,20],[6,130],[111,130],[111,20]]).astype(np.float32)
    warp_mat1 = cv2.getPerspectiveTransform(src1,dst1)
    warp_dst1 = cv2.warpPerspective(im1, warp_mat1, (im1.shape[1], im1.shape[0]))

    src2 = np.array([[181, 5], [120, 50], [176, 120], [249, 70]]).astype(np.float32)
    warp_mat2 = cv2.getPerspectiveTransform(src2, dst1)
    warp_dst2 = cv2.warpPerspective(im1, warp_mat2, (im1.shape[1], im1.shape[0]))

    src3 = np.array([[78, 163], [133, 244], [245, 160], [146,117]]).astype(np.float32)
    warp_mat3 = cv2.getPerspectiveTransform(src3, dst1)
    warp_dst3 = cv2.warpPerspective(im1, warp_mat3, (im1.shape[1], im1.shape[0]))

    warp_dst1 = clean_SP_noise_single(warp_dst1, 3)
    warp_dst2 = clean_SP_noise_single(warp_dst2, 3)
    warp_dst3 = clean_SP_noise_single(warp_dst3, 3)

    ls = [warp_dst3,warp_dst1,warp_dst2]
    clean_im = clean_SP_noise_multiple(ls)
    return clean_im


def clean_windmill(im):
    x = 125.0
    xx = 132.0
    y = 100.0
    yy = 155.0

    ffIm = np.fft.fft2(im)
    ffShiftedIm = np.fft.fftshift(ffIm)
    val = 1

    clean_im = np.fft.ifft2(np.fft.ifftshift(np.array([[1 if math.sqrt((i - x) ** 2 + (j - y) ** 2) > val and
                                                             math.sqrt((i - xx) ** 2 + (j - yy) ** 2) > val else
                                                        0 for j in range(len(im[0]))] for i in
                                                       range(len(im))]) * np.fft.fftshift(np.fft.fft2(im)))).astype(int)
    return clean_im


def clean_watermelon(im):

    kernel = np.array([[0, -1, 0],
                       [-1, 5.1, -1],
                       [0, -1, 0]])

    clean_im = cv2.filter2D(im, -1, kernel)

    return clean_im


def clean_umbrella(im):

    matrix = np.zeros([256,256])
    matrix[0][0] = 0.5
    matrix[4][79] = 0.5
    matrix = np.fft.fft2(matrix)
    im_fft = np.fft.fft2(im)
    matrix[abs(matrix) < 0.02] = 1
    im_fft = im_fft / matrix

    clean_im = abs(np.fft.ifft2(im_fft))
    # im_show(clean_im)

    return clean_im


def clean_USAflag(im):
    clean_im = clean_SP_noise_single(im,3)
    clean_im[1: 90, 1: 150] = im[1: 90, 1: 150]

    return clean_im


def clean_cups(im):
    f = np.fft.fft2(im)
    shifted = np.fft.fftshift(f)
    indexx = 109
    indexx2 = 147
    indexy = 107
    indexy2 = 148
    for i in range(indexx, indexx2):
        for j in range(indexy, indexy2):
            shifted[i][j] *= 1.5
    shifted = np.fft.ifftshift(shifted)
    clean_im = abs(np.fft.ifft2(shifted))

    return clean_im


def dirac_delta(n, N, q):
    return 1 if n == q else 0


def clean_house(im):
    sum = 0
    for j in range(0, 10):
        x = np.array([dirac_delta(n, im.shape[1], j) for n in range(im.shape[1])])
        X = np.fft.fft(x)
        sum += X

    im_fft = np.fft.fft2(im)

    im_fft = im_fft / (sum * 0.1)

    clean_im =abs( np.fft.ifft2(im_fft))
    # im_show(clean_im)
    clean_im = np.clip(clean_im,0,255)
    return clean_im


def clean_bears(im):
    clean_im = cv2.equalizeHist(im)
    contrast_factor = 0.5
    clean_im = cv2.addWeighted(clean_im, contrast_factor, clean_im, 0, 0)
    return clean_im

#
# def im_show(img):
#     # an example of how to use fourier transform:
#     # img = cv2.imread(im_name)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     img_fourier = np.fft.fft2(img)  # fft - remember this is a complex numbers matrix
#     img_fourier = np.fft.fftshift(img_fourier)  # shift so that the DC is in the middle
#
#     plt.figure()
#     plt.subplot(1, 3, 1)
#     plt.imshow(img, cmap='gray')
#     plt.title('original image')
#
#     plt.subplot(1, 3, 2)
#     plt.imshow(np.log(abs(img_fourier)),
#                cmap='gray')  # need to use abs because it is complex, the log is just so that we can see the difference in values with out eyes.
#     plt.title('fourier transform of image')
#
#     img_inv = np.fft.ifft2(img_fourier)
#     plt.subplot(1, 3, 3)
#     plt.imshow(abs(img_inv), cmap='gray')
#     plt.title('inverse fourier of the fourier transform')
