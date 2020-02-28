# -*- coding:UTF-8 -*-
"""
This file used for generateing the HOG features of an image
the HOG feature can be used for classifiers to judge whether
an image has text
"""
import numpy as np
import cv2 as cv
from typing import List, Tuple


def normalize_img(gray_img: np.ndarray) -> np.ndarray:
    """
    This function will normalize the gray level into scale 0-1
    Inputs:
        gray_img: the gray image
    Output:
        the normalized result of the image
    """
    h, w = gray_img.shape
    for i in range(h):
        for j in range(w):
            gray_img[i, j] = (gray_img[i, j]+0.5)/256
    return gray_img


def gamma_calibrate(gray_img: np.ndarray, gamma_val=2.0) -> np.ndarray:
    """
    Calibrating the effects of the lightness
    Inputs:
        gray_img: the gray image
        gamma_val: the gamma value to adjust the gray value of image
    Output:
        the calidrated gray image
    """
    h, w = gray_img.shape
    exp_val = 1/gamma_val
    for i in range(h):
        for j in range(w):
            gray_img[i, j] = np.math.pow(gray_img[i, j], exp_val)
    return gray_img


def normalize_gamma_recover(gray_img: np.ndarray) -> np.ndarray:
    """
    After the normalizing and gamma calibration, the gray image
    needs to recover from 0-1 range to 0-255 range
    Input:
        gray_img: the gray image processed with normalizing and gamma
    Output:
        the recovered gray image with gray value from 0-255
    """
    h, w = gray_img.shape
    for i in range(h):
        for j in range(w):
            gray_img[i, j] = gray_img[i, j]*256-0.5
    gray_img = np.array(gray_img, dtype="uint8")
    return gray_img


def pixel_desc(cell_matrix: np.ndarray, degree_bin_size=30) -> Tuple[int, float]:
    """
    Calculating the gradient and the orientation of a pixel
    the orientation will be in degree format, each 
        'degree_bin_size' in a bin
    Inputs:
        cell_matrix: the matrix with size 3*3, cover the neightbor
        degree_bin_size: each n degree will be assigned in a bin
    Output:
        bin_idx: the degree of the gradient assigned in a bin with bin
                    size = degree_bin_size
        gradient_val: the gradient value of the pixel
    """
    assert cell_matrix.shape == (3, 3)
    assert 360 % degree_bin_size == 0

    gx = int(cell_matrix[2, 1])-int(cell_matrix[0, 1])
    gy = int(cell_matrix[1, 2])-int(cell_matrix[1, 0])
    gradient_val = np.math.sqrt(gx**2 + gy**2)
    degree = np.math.atan2(gy, gx)
    degree = np.math.degrees(degree)
    degree = degree if degree >= 0 else (360 + degree)
    bin_idx = int(degree/degree_bin_size)
    return bin_idx, gradient_val


def _check_bound(i: int, j: int, x_low: int, x_high: int, y_low: int, y_high: int) -> bool:
    """
    check whether a position is valid
    """
    if i < y_low or i >= y_high or j < x_low or j >= x_high:
        return False
    else:
        return True


def cell_desc(gray_img: np.ndarray, cell_x_low: int, cell_x_high: int,
              cell_y_low: int, cell_y_high: int, degree_bin_size=30) -> List[float]:
    """
    Calculating the descriptor of a cell
    Inputs:
        gray_img: a gray level image
        cell**: the x and y index range in the gray image, high is excluded
        degree_bin_size: each bin contains a range of degree
    Output:
        the descriptor of the cell
    """
    assert cell_x_high-cell_x_low == cell_y_high-cell_y_low
    assert 360 % degree_bin_size == 0

    h, w = gray_img.shape
    descs: List[float] = [0.0]*(360//degree_bin_size)
    for i in range(cell_y_low, cell_y_high):
        for j in range(cell_x_low, cell_x_high):
            neighbor_mat = np.zeros([3, 3], dtype="int")
            neighbor_mat[0, 0] = gray_img[i-1, j-1] \
                if _check_bound(i-1, j-1, 0, w, 0, h) else 0
            neighbor_mat[0, 1] = gray_img[i-1, j] \
            if _check_bound(i-1, j, 0, w, 0, h) else 0
            neighbor_mat[0, 2] = gray_img[i-1, j+1] \
            if _check_bound(i-1, j+1, 0, w, 0, h) else 0
            neighbor_mat[1, 0] = gray_img[i, j-1] \
            if _check_bound(i, j-1, 0, w, 0, h) else 0
            neighbor_mat[1, 1] = gray_img[i, j] \
            if _check_bound(i, j, 0, w, 0, h) else 0
            neighbor_mat[1, 2] = gray_img[i, j+1] \
            if _check_bound(i, j+1, 0, w, 0, h) else 0
            neighbor_mat[2, 0] = gray_img[i+1, j-1] \
            if _check_bound(i+1, j-1, 0, w, 0, h) else 0
            neighbor_mat[2, 1] = gray_img[i+1, j] \
            if _check_bound(i+1, j, 0, w, 0, h) else 0
            neighbor_mat[2, 2] = gray_img[i+1, j+1] \
            if _check_bound(i+1, j+1, 0, w, 0, h) else 0

            bin_idx, g_val = pixel_desc(neighbor_mat)
            descs[bin_idx] += g_val
    return descs


def block_desc(gray_img: np.ndarray, block_x_low: int, block_x_high: int,
               block_y_low: int, block_y_high: int, cell_size=6) -> List[List[float]]:
    """
    Calculating the descriptors of a block
    Inputs:
        gray_img: gray level image
        block_**: the index range of the block in gray image, high is excluded
        cell_size: the cell is 'cell_size*cell_size' in the block
    Output:
        the descriptors of the cells in the block wll be concatenated
    """
    assert block_x_high-block_x_low == block_y_high-block_y_low
    assert (block_x_high-block_x_low) % cell_size == 0

    block_desc: List[List[float]] = []
    for i in range(block_y_low, block_y_high, cell_size):
        for j in range(block_x_low, block_x_high, cell_size):
            block_desc.append(cell_desc(gray_img, cell_x_low=j, cell_x_high=j+cell_size,
                                        cell_y_low=i, cell_y_high=i+cell_size))
    return block_desc


def gray_img_desc(gray_img: np.ndarray, block_cell_size=3, cell_size=6) -> List[List[List[float]]]:
    """
    Calculating the descriptor of a gray image
    Inputs:
        gray_img: the gray image
        block_cell_size: the cell number in one block is 'block_cell_size*block_cell_size'
        cell_size: the size of one cell is 'cell_size*cell_size' pixels
    Output:
        the descriptor of the whole gray image
    """
    h, w = gray_img.shape
    block_pixel_per_dim = block_cell_size*cell_size
    img_desc: List[List[List[float]]] = []
    for i in range(0, h, block_pixel_per_dim):
        for j in range(0, w, block_pixel_per_dim):
            img_desc.append(
                block_desc(gray_img, j, j+block_pixel_per_dim,
                           i, i+block_pixel_per_dim, cell_size=cell_size)
            )
    return img_desc


if __name__ == "__main__":
    img = cv.imread("breaking_news.png")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(300,500))
    img = np.asarray(img, dtype="float")
    img = normalize_img(img)
    img = gamma_calibrate(img)
    img = normalize_gamma_recover(img)
    # descriptor
    img_desc = gray_img_desc(img)
    img_desc=np.array(img_desc,dtype="float")

