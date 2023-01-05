#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
import glob
import random


def rotate_90(image, label, angle, scale=1):
    h, w, _ = image.shape
    # rotate matrix
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    # rotate
    image = cv2.warpAffine(image, M, (w, h))
    label = cv2.warpAffine(label, M, (w, h))
    return image, label


def crop_and_resize(image, label):
    h, w, _ = image.shape
    s = random.uniform(0.55,2.0)
    nh, nw = int(h * s), int(w * s)

    img = cv2.resize(image, (nh, nw), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(label, (nh, nw), interpolation=cv2.INTER_NEAREST)

    y = random.randint(0, nh - h - 1)
    x = random.randint(0, nw - w - 1)
    cropped_image = img[y:y + h, x:x + w]
    cropped_label = lab[y:y + h, x:x + w]
    return cropped_image, cropped_label


def resize(image, label):
    h, w, _ = image.shape
    s = random.uniform(0.51, 2.0)

    # nh, nw = int(h * s), int(w * s)
    # img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # lab = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

    img = cv2.resize(image, dsize=(0,0),fx=s,fy=s, interpolation=cv2.INTER_CUBIC)
    lab = cv2.resize(label, dsize=(0,0),fx=s,fy=s, interpolation=cv2.INTER_NEAREST)

    return img, lab


def addGaussianNoise(image, percetage):
    h, w, _ = image.shape
    G_Noiseimg = image.copy()
    G_NoiseNum = int(percetage * w * h)
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0, h)
        temp_y = np.random.randint(0, w)
        G_Noiseimg[temp_x][temp_y] = np.random.randn(1)[0]
    return G_Noiseimg


def add_noise(image):
    h, w, _ = image.shape
    for i in range(np.random.randint(20000)):  # 添加点噪声
        temp_x = np.random.randint(0, w)
        temp_y = np.random.randint(0, h)
        image[temp_y][temp_x] = [255, 255, 255]
    return image


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def darker(image, percetage=np.random.randint(60, 95) / 100):
    imgarr = np.array(image, np.uint8)
    # get darker
    image_copy = np.clip((imgarr * percetage).astype(np.uint8), a_max=255, a_min=0)

    return image_copy


def brighter(image, percetage=np.random.randint(105, 115) / 100):
    imgarr = np.array(image, np.uint8)
    # get darker
    image_copy = np.clip((imgarr * percetage).astype(np.uint8), a_max=255, a_min=0)
    return image_copy


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def rotate(img, label, limit_up=45, limit_down=45):
    # 旋转矩阵
    rows, cols = img.shape[:2]
    center_coordinate = (int(cols / 2), int(rows / 2))
    angle = random.uniform(limit_down, limit_up)
    M = cv2.getRotationMatrix2D(center_coordinate, angle, 1)

    # 仿射变换
    out_size = (cols, rows)
    rotate_img = cv2.warpAffine(img, M, out_size)
    rotate_label = cv2.warpAffine(label, M, out_size, flags=cv2.INTER_NEAREST, borderValue=255)

    return rotate_img, rotate_label


def shift(img, label, distance_down=-100, distance_up=100):
    rows, cols = img.shape[:2]
    y_shift = random.uniform(distance_down, distance_up)
    x_shift = random.uniform(distance_down, distance_up)

    # 生成平移矩阵
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    # 平移
    img_shift = cv2.warpAffine(img, M, (cols, rows))
    # label_shift = cv2.warpAffine(label, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
    #                              borderValue=255)
    label_shift = cv2.warpAffine(label, M, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)

    return img_shift, label_shift


def saturation_jitter(cv_img, jitter_range):
    """
    调节图像饱和度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        饱和度调整后的图像
    """

    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def brightness_jitter(cv_img, jitter_range):
    """
    调节图像亮度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        亮度调整后的图像
    """

    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1.0 - jitter_range)
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def contrast_jitter(cv_img, jitter_range):
    """
    调节图像对比度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        对比度调整后的图像
    """

    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(greyMat)
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def random_jitter(cv_img, saturation_range, brightness_range, contrast_range):
    """
    图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果

    Args:
        cv_img(numpy.ndarray): 输入图像
        saturation_range(float): 饱和对调节范围，0-1
        brightness_range(float): 亮度调节范围，0-1
        contrast_range(float): 对比度调节范围，0-1

    Returns:
        亮度、饱和度、对比度调整后图像
    """

    saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
    brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
    contrast_ratio = np.random.uniform(-contrast_range, contrast_range)

    order = [0, 1, 2]
    np.random.shuffle(order)

    for i in range(3):
        if order[i] == 0:
            cv_img = saturation_jitter(cv_img, saturation_ratio)
        if order[i] == 1:
            cv_img = brightness_jitter(cv_img, brightness_ratio)
        if order[i] == 2:
            cv_img = contrast_jitter(cv_img, contrast_ratio)
    return cv_img


def add_gasuss_noise(image, mean=0, var=0.003):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def augment(img, label):
    if np.random.random() < 0.20:
        img, label = rotate_90(img, label, 90)
    if np.random.random() < 0.20:
        img, label = rotate_90(img, label, 180)
    if np.random.random() < 0.20:
        img, label = rotate_90(img, label, 270)
    if np.random.random() < 0.30:
        img = cv2.flip(img, 1)  # flipcode = 1：沿y轴翻转
        label = cv2.flip(label, 1)
    if np.random.random() < 0.30:
        img = random_jitter(img, 0.1, 0.1, 0.1)

    # if np.random.random() < 0.2:
    #     img, label = crop_and_resize(img, label)
        # img, label = resize(img, label)
    # if np.random.random() < 0.10:
    #     img = blur(img)
    # if np.random.random() < 0.10:
    #     img = add_gasuss_noise(img)
    #
    # if np.random.random() < 0.20:
    #     img, label = rotate(img, label, 10, -10)
    # if np.random.random() < 0.20:
    #     img, label = shift(img, label, -20, 20)

    # if np.random.random() < 0.15:
    #     img = brighter(img)
    # if np.random.random() < 0.15:
    #     img = darker(img)
    # if np.random.random() < 0.10:
    #     img = random_gamma_transform(img, 1.0)
    # if np.random.random() < 0.10:
    #     img = add_noise(img)

    return img, label


if __name__ == "__main__":
    img_path = '/home/data1/jojolee/Tianzhibei2020/Tianzhibei2019OriginalDataset/cropped/512/valid/img/1.tif'
    label_path = '/home/data1/jojolee/Tianzhibei2020/Tianzhibei2019OriginalDataset/cropped/512/valid/label_color/1.png'
    image = cv2.imread(img_path)
    label = cv2.imread(label_path)
    cv2.imshow('image', image)
    cv2.imshow('label', label)
    img, lab = augment(image, label)
    cv2.imshow('img', img)
    cv2.imshow('lab', lab)
    cv2.waitKey()
