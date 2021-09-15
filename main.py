import numpy as np
import cv2


def creat_first_patch(image, patch_size=50, overlap_size=10, size=2500):
    n = np.ceil((size-patch_size)/(patch_size-overlap_size))
    size = (patch_size + (patch_size-overlap_size)*n).astype('int')
    image_out = np.uint8(np.zeros([size, size, 3]))
    x, y = np.random.randint(0, image.shape[0] - patch_size), np.random.randint(0, image.shape[1] - patch_size)
    image_out[0:patch_size, 0:patch_size, :] = image[x:x + patch_size, y:y + patch_size, :]
    return image_out, size


def find_a_patch_in_row(patch, image, patch_size=50, overlap_size=10, trh=0.99):
    matched = cv2.matchTemplate(image, patch[0:patch_size,
                                       patch_size - overlap_size:patch_size, :], cv2.TM_CCOEFF_NORMED)
    matched = matched[0:image.shape[0] - patch_size, 0:image.shape[1] - patch_size]
    x, y = (np.where(matched >= trh * np.max(matched)))
    random = np.random.randint(len(x))
    next_patch = np.copy(image[x[random]:x[random] + patch_size, y[random]:y[random] + patch_size, :])
    return next_patch



def find_a_patch_in_column(patch, image, patch_size=50, overlap_size=10, trh=0.99):
    matched = cv2.matchTemplate(image, patch[patch_size - overlap_size:patch_size,
                                                0:patch_size, :], cv2.TM_CCOEFF_NORMED)
    matched = matched[0:image.shape[0] - patch_size, 0:image.shape[1] - patch_size]
    x, y = (np.where(matched >= trh * np.max(matched)))
    random = np.random.randint(len(x))
    next_patch = np.copy(image[x[random]:x[random] + patch_size, y[random]:y[random] + patch_size, :])
    return next_patch


def find_a_patch(patch, image, patch_size=50, overlap_size=10, trh=0.9):
    patch[overlap_size:, overlap_size:, 0] = 255
    # patch = patch[0:overlap_size, :, :]
    matched = cv2.matchTemplate(image, patch, cv2.TM_CCOEFF_NORMED)
    matched = matched[0:image.shape[0] - patch_size, 0:image.shape[1] - patch_size]
    x, y = (np.where(matched >= trh * np.max(matched)))
    random = np.random.randint(len(x))
    # print(x[random], y[random])
    next_patch = np.copy(image[x[random]:x[random] + patch_size, y[random]:y[random] + patch_size, :])
    return next_patch


def min_cut(array, type='column'):
    if type == 'row':
        array = np.transpose(array)

    path = np.zeros(array.shape).astype('int')
    cost_j = array[:, 0]

    for j in range(1, array.shape[1]):
        cost_j_plus = array[:, j]
        new_cost = np.inf * np.ones(array.shape[0]).astype('int')
        for i in range(array.shape[0]):
            for di in ([-1, 0, 1]):
                if (i+di > -1) & (i+di < array.shape[0]):
                    if cost_j[i+di]+cost_j_plus[i] < new_cost[i]:
                        new_cost[i] = cost_j[i+di]+cost_j_plus[i]
                        path [i, j] = i+di
        cost_j = new_cost

    bounds = np.zeros(array.shape[1]).astype('int')
    bounds[-1] = np.where(cost_j == np.max(cost_j))[0][0]
    for i in range (len(bounds)-1, 0, -1):
        bounds[i-1] = path[bounds[i], i]
    return bounds

def fill_first_row(image, image_out, overlap_size=10,
                   patch_size=50, size=2500, trh=0.99):
    num_of_patch = np.ceil((size - patch_size) / (patch_size - overlap_size)).astype('int')
    for i in range(num_of_patch):
        patch = image_out[0:patch_size, i * (patch_size - overlap_size):i * (patch_size - overlap_size)
                                                                        + patch_size, :]
        next_patch = find_a_patch_in_row(np.copy(patch), image, patch_size, overlap_size, trh=trh)
        A = patch[0:patch_size, patch_size - overlap_size:patch_size, :].astype('int')
        B = next_patch[0:patch_size, 0:overlap_size, :].astype('int')
        array = (A - B) ** 2
        array = array[:, :, 0] + array[:, :, 1] + array[:, :, 2]
        bounds = min_cut(array, type='row')
        for j in range(patch_size):
            next_patch[j, 0:bounds[j], :] = patch[j, patch_size-overlap_size:patch_size-overlap_size+bounds[j], :]
        if patch_size + (i + 1) * (patch_size - overlap_size) <= size:
            image_out[0: patch_size,
            (i + 1) * (patch_size - overlap_size):patch_size + (i + 1) * (patch_size - overlap_size),
            :] = next_patch
        else:
            image_out[0: patch_size,
            (i + 1) * (patch_size - overlap_size):, :] = \
                next_patch[:, 0:2500 - patch_size - (i + 1) * (patch_size - overlap_size), :]
    return image_out


def fill_first_column(image, image_out, overlap_size=10,
                   patch_size=50, size=2500, trh=0.99):
    num_of_patch = np.ceil((size - patch_size) / (patch_size - overlap_size)).astype('int')
    for i in range(num_of_patch):
        patch = image_out[i * (patch_size - overlap_size):i * (patch_size - overlap_size)+ patch_size,
                                                    0:patch_size, :]
        next_patch = find_a_patch_in_column(patch, image, patch_size, overlap_size, trh=trh)
        A = patch[patch_size - overlap_size:patch_size, 0:patch_size, :].astype('int')
        B = next_patch[0:overlap_size, 0:patch_size, :].astype('int')
        array = (A - B) ** 2
        array = array[:, :, 0] + array[:, :, 1] + array[:, :, 2]
        bounds = min_cut(array)
        for j in range(patch_size):
            next_patch[0:bounds[j], j, :] = patch[patch_size-overlap_size:patch_size-overlap_size+bounds[j], j, :]

        if patch_size + (i + 1) * (patch_size - overlap_size) <= size:
            image_out[(i + 1) * (patch_size - overlap_size):patch_size + (i + 1) * (patch_size - overlap_size),
                                                    0: patch_size, :] = next_patch
        else:
            image_out[(i + 1) * (patch_size - overlap_size):, 0: patch_size, :] = \
                next_patch[0:2500 - patch_size - (i + 1) * (patch_size - overlap_size), :, :]
    return image_out


def fill_image(image, image_out, overlap_size=10,
                   patch_size=50, size=2500, trh = 0.99):
    num_of_patch = np.ceil((size - patch_size) / (patch_size - overlap_size)).astype('int')
    for col in range(1, num_of_patch+1):
        print(col)
        for row in range(1, num_of_patch+1):
            patch = image_out[col * (patch_size - overlap_size):col * (patch_size - overlap_size) + patch_size,
                                row * (patch_size - overlap_size):row * (patch_size - overlap_size) + patch_size, :]
            next_patch = find_a_patch(np.copy(patch), image, patch_size, overlap_size, trh=trh)

            A = patch[patch_size - overlap_size:patch_size, 0:patch_size, :].astype('int')
            B = next_patch[0:overlap_size, 0:patch_size, :].astype('int')
            array = (A - B) ** 2
            array = array[:, :, 0] + array[:, :, 1] + array[:, :, 2]
            bounds = min_cut(array)
            for j in range(patch_size):
                next_patch[0:bounds[j], j, :] = patch[0: bounds[j], j, :]
            A = patch[0:patch_size, patch_size - overlap_size:patch_size, :].astype('int')
            B = next_patch[0:patch_size, 0:overlap_size, :].astype('int')
            array = (A - B) ** 2
            array = array[:, :, 0] + array[:, :, 1] + array[:, :, 2]
            bounds = min_cut(array, type='row')
            for j in range(patch_size):
                next_patch[j, 0:bounds[j], :] = patch[j, 0:bounds[j], :]

            if (patch_size + (col) * (patch_size - overlap_size) <= size) & \
                    (patch_size + (row) * (patch_size - overlap_size) <= size):
                image_out[(col) * (patch_size - overlap_size):patch_size + (col) * (patch_size - overlap_size),
                (row) * (patch_size - overlap_size):patch_size + (row) * (patch_size - overlap_size), :] =\
                    next_patch

            elif (patch_size + (col) * (patch_size - overlap_size) > size) & \
                    (patch_size + (row) * (patch_size - overlap_size) <= size):
                image_out[(col) * (patch_size - overlap_size):,
                (row) * (patch_size - overlap_size):patch_size + (row) * (patch_size - overlap_size), :] = \
                    next_patch[0:2500 - patch_size - (col) * (patch_size - overlap_size), :, :]

            elif (patch_size + (col) * (patch_size - overlap_size) <= size) & \
                    (patch_size + (row) * (patch_size - overlap_size) > size):
                image_out[(col) * (patch_size - overlap_size):patch_size + (col) * (patch_size - overlap_size),
                (row) * (patch_size - overlap_size):, :] = \
                    next_patch[:, 0:2500 - patch_size - (row) * (patch_size - overlap_size), :]
                print('     =', row)
            else:
                image_out[(col) * (patch_size - overlap_size):, (row) * (patch_size - overlap_size):, :] = \
                    next_patch[0:2500 - patch_size - (col) * (patch_size - overlap_size),
                    0:2500 - patch_size - (row) * (patch_size - overlap_size), :]

    return  image_out


img = cv2.imread('texture1.jpg')
patch_size, overlap_size = 150, 50
img_out, size = creat_first_patch(img, patch_size=patch_size, overlap_size=overlap_size)
img_out = fill_first_row(img, img_out, overlap_size=overlap_size, patch_size=patch_size, trh=0.5, size=size)
img_out = fill_first_column(img, img_out, overlap_size=overlap_size, patch_size=patch_size, trh=0.5, size=size)
img_out = fill_image(img, img_out, overlap_size=overlap_size, patch_size=patch_size, trh=0.5, size=size)
cv2.imwrite('res1.1.jpg', img_out[0:2500, 0:2500, :])
img_out = cv2.blur(img_out, (6, 6))
cv2.imwrite('res1.jpg', img_out[0:2500, 0:2500, :])
