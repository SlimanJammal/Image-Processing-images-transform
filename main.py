from transforms import *


if __name__ == "__main__":

    print("-----------------------image 1----------------------\n")
    im1 = cv2.imread(r'Images\baby.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1_clean = clean_baby(im1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im1_clean, cmap='gray', vmin=0, vmax=255)
    #
    print("-----------------------image 2----------------------\n")
    im2 = cv2.imread(r'Images\windmill.tif')
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im2_clean = clean_windmill(im2)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im2_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 3----------------------\n")
    im3 = cv2.imread(r'Images\watermelon.tif')
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    im3_clean = clean_watermelon(im3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im3_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 4----------------------\n")
    im4 = cv2.imread(r'Images\umbrella.tif')
    im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2GRAY)
    im4_clean = clean_umbrella(im4)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im4, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im4_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 5----------------------\n")
    im5 = cv2.imread(r'Images\USAflag.tif')
    im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2GRAY)
    im5_clean = clean_USAflag(im5)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im5, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im5_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 6----------------------\n")
    im6 = cv2.imread(r'Images\cups.tif')
    im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2GRAY)
    im6_clean = clean_cups(im6)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im6, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im6_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 7----------------------\n")
    im7 = cv2.imread(r'Images\house.tif')
    im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2GRAY)
    im7_clean = clean_house(im7)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im7, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im7_clean, cmap='gray', vmin=0, vmax=255)


    print("-----------------------image 8----------------------\n")
    im8 = cv2.imread(r'Images\bears.tif')
    im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2GRAY)
    im8_clean = clean_bears(im8)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im8, cmap='gray', vmin=0, vmax=255)
    plt.subplot(1, 2, 2)
    plt.imshow(im8_clean, cmap='gray', vmin=0, vmax=255)


plt.show()




























#
# x_ = 116
# x__ = 242
# y_ = 77
# y__ = 244
#
# image2 = clean_SP_noise_single(im[x_: x__, y_: y__], 1)
#
# points1 = np.float32([[20, 6, 1],
#                       [21, 111, 1],
#                       [129, 111, 1],
#                       [130, 6, 1]])
#
# x_, x__, y_, y__ = clean_baby_helper(points1)
# fina1 = clean_SP_noise_single(im[x_: x__, y_: y__], 1)
#
# points2 = np.float32([[46, 0, 1],
#                       [0, 69, 1],
#                       [45, 166, 1],
#                       [124, 55, 1]])
# points3 = np.float32([[0, 0, 1],
#                       [0, 104, 1],
#                       [109, 104, 1],
#                       [109, 0, 1]])
# transform = find_transform(points2, points3)
# final2 = trasnform_image(image2, transform)
#
# points4 = np.float32([[3, 180, 1],
#                       [69, 250, 1],
#                       [119, 176, 1],
#                       [50, 121, 1]])
# x_, x__, y_, y__ = clean_baby_helper(points4)
# image3 = clean_SP_noise_single(im[x_: x__, y_: y__], 1)
# final3 = trasnform_image(image3, transform)
# clean_im = np.zeros(fina1.shape)
# for i in range(len(fina1)):
#     for j in range(len(fina1[0])):
#         clean_im[i][j] = np.median([fina1[i][j], final2[i][j], final3[i][j]])
#
# clean_im = clean_SP_noise_single(clean_im, 3)
# return clean_im