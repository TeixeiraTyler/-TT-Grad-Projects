
import numpy
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import torchvision
from torchvision import datasets, transforms
import cv2
from skimage import data, filters
import sys


# function to produce a gaussian kernel given expected size and standard deviation.
def get_gaussian_kernel(size, sigma, direction):
    size = int(size) // 2
    axis = direction
    x, y = numpy.mgrid[-size:size+1, 0:1]
    normal = 1/(2.0*numpy.pi*sigma**2)
    gaussian = numpy.exp(-((x**2+y**2)/(2.0*sigma**2))) * normal
    return gaussian


# function to compute gradient in a particular direction.
def directional_filter(img, axis):
    if (axis == 'x'):
        x = torch.tensor([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
        newimage = ndimage.convolve(img, x)
    if (axis == 'y'):
        y = torch.tensor([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]])
        newimage = ndimage.convolve(img, y)
    return newimage


# function to perform non-max suppression.
def nonmaximum_suppression(img, direction):
    angle = direction * 180. / numpy.pi
    angle[angle < 0] += 180
    x, y = img.shape
    supImage = numpy.zeros((x, y), dtype=numpy.int32)

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            a = 40
            b = 40

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                a = img[i, j + 1]
                b = img[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                a = img[i + 1, j - 1]
                b = img[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                a = img[i + 1, j]
                b = img[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                a = img[i - 1, j - 1]
                b = img[i + 1, j + 1]

            if (img[i, j] >= a) and (img[i, j] >= b):
                supImage[i, j] = img[i, j]
            else:
                supImage[i, j] = 0

    return supImage


# STEP 1: This is the original grey-scale image saved as a matrix.
image = imageio.imread("image3.jpg")
image = image.astype('int32')
plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.show()

# STEP 2: Create mask 'G' with sigma equal to input argument.
kernelSize = 3
deviation = float(sys.argv.pop())  # get sigma from system
Gmask = get_gaussian_kernel(kernelSize, deviation, None)
G = ndimage.convolve(image, Gmask)
gaussianImage = ndimage.gaussian_filter(image, sigma=deviation)
plt.imshow(gaussianImage, cmap=plt.get_cmap('gray'))
plt.show()

# STEP 3: Create directional masks Gx and Gy
Gx = get_gaussian_kernel(kernelSize, deviation, 'x')
Gy = get_gaussian_kernel(kernelSize, deviation, 'y')

# STEP 4: Convolve directional masks Gx and Gy over gaussian image.
Ximage = directional_filter(gaussianImage, 'x')
plt.imshow(Ximage, cmap=plt.get_cmap('gray'))
plt.show()

Yimage = directional_filter(gaussianImage, 'y')
plt.imshow(Yimage, cmap=plt.get_cmap('gray'))
plt.show()

# STEP 5: Convolve Ix with Gx and Iy with Gy
xcomponent = ndimage.convolve(Ximage, Gx)
ycomponent = ndimage.convolve(Yimage, Gy)

# STEP 6: Magnitude
m = numpy.degrees(numpy.arctan2(Yimage, Ximage))
magnitude = numpy.hypot(Ximage, Yimage)
plt.imshow(magnitude, cmap=plt.get_cmap('gray'))
plt.show()

# STEP 7: Non-maximum suppression.
nms = nonmaximum_suppression(magnitude, m)
plt.imshow(nms, cmap=plt.get_cmap('gray'))
plt.show()

# STEP 8: Hysteresis thresholding.
low = 0.1
high = 0.35
final = filters.apply_hysteresis_threshold(nms, low, high)
plt.imshow(final, cmap=plt.get_cmap('gray'))
plt.show()

image = Image.open("image.jpg")
canny = cv2.Canny(image, 60, 150)

titles = ['image', 'canny']
images = [image, canny]

for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()



#
# print(image.format, image.size, image.mode)
#
#
#
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

