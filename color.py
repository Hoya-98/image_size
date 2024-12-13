from typing import List, Union, Optional

import numpy as np
from numpy.typing import NDArray
import cv2

#from colour import colour_correction
from libs.colour.colour.characterisation.correction import colour_correction


COLOR_CONFIGS = {
    'color_references': [
        [230, 202, 183], [199, 168, 145], [199, 159, 145],
        [199, 161, 146], [163, 123, 94], [195, 162, 141],
        [176, 62, 60], [68, 142, 77], [42, 76, 149]
    ],
    'matrix': {
        'rgb2yuv': [
            [0.257, 0.504, 0.098],
            [-0.148, -0.291, 0.439],
            [0.439, -0.368, -0.071]
        ],
        'yuv2rgb': [
            [1.164, 0, 1.596],
            [1.164, -0.813, -0.391],
            [1.164, 2.018, 0]
        ]
    }
}

COLOR_REFERENCES = COLOR_CONFIGS['color_references']
MATRIX_RGB2YUV = COLOR_CONFIGS['matrix']['rgb2yuv']
MATRIX_YUV2RGB = COLOR_CONFIGS['matrix']['yuv2rgb']


def transform_with_matrix(array_like: Union[List, NDArray], transform_str: str) -> NDArray:
    if transform_str == 'rgb2yuv':
        transform_matrix = MATRIX_RGB2YUV
    elif transform_str == 'yuv2rgb':
        transform_matrix = MATRIX_YUV2RGB
    else:
        raise NotImplementedError
    out = np.dot(array_like, transform_matrix)

    return out + [16, 128, 128] if transform_str == 'rgb2yuv' else out  # in this part, the type is auto-casted...


def transform_rgb2yuv(rgb_array: Union[List, NDArray]) -> NDArray:
    # return np.dot(rgb_array, MATRIX_RGB2YUV) + [16, 128, 128]
    return transform_with_matrix(rgb_array, 'rgb2yuv')


def transform_yuv2rgb(yuv_array: Union[List, NDArray]) -> NDArray:
    # return np.dot(yuv_array, MATRIX_YUV2RGB)
    return transform_with_matrix(yuv_array, 'yuv2rgb')


def adjust_colors(
        img: NDArray, marker_img: NDArray,
        marker_type: int, ccm_number: int,
        color_mode: str = 'RGB', weights: Optional[List] = None
) -> NDArray:
    yuv_flag = color_mode == 'YUV'
    weights = [1, 1, 1] if weights is None else weights

    marker_type_dict = {
        'height_interval': {
            1: np.array([2.5, 9.5, 16.5]),
            2: np.array([5, 17, 29]),
            3: np.array([7.5, 24.5, 41.5]),
            4: np.array([10, 32, 54])
        },
        'width_interval': {
            1: np.array([9.5, 16.5, 23.5]),
            2: np.array([17, 29, 41]),
            3: np.array([24.5, 41.5, 58.5]),
            4: np.array([32, 54, 76])
        }, 'marker_h_w': {1: (19, 33), 2: (34, 58), 3: (49, 83), 4: (64, 108)}
    }
    correction_method_dict = {1: 'Cheung 2004', 2: 'Finlayson 2015', 3: 'Vandermonde'}
    # method : Cheung 2004, Finlayson 2015, Vandermonde

    mk_img_height, mk_img_width = marker_img.shape[0], marker_img.shape[1]

    if marker_type in range(1, 5):
        mk_height_interval = marker_type_dict['height_interval'][marker_type]
        mk_width_interval = marker_type_dict['width_interval'][marker_type]
        mk_height, mk_width = marker_type_dict['marker_h_w'][marker_type]

    else:
        raise ValueError('The marker type is not defined correctly; must be in a closed integer interval [1, 4]')

    mk_pos_height = (mk_height_interval / mk_height * mk_img_height).astype(int)
    mk_pos_width = (mk_width_interval / mk_width * mk_img_width).astype(int)

    src = list()
    roi_height = mk_img_height // 15
    roi_width = mk_img_width // 25
    # for height, width in zip(mk_pos_height, mk_pos_width):
    for height in mk_pos_height:
        for width in mk_pos_width:
            h_neg, h_pos = height - roi_height, height + roi_height
            w_neg, w_pos = width - roi_width, width + roi_width
            color = cv2.mean(marker_img[h_neg:h_pos, w_neg:w_pos])
            src.append(color[:3])

    img_ori = img
    if yuv_flag:
        img_ori = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        src = transform_rgb2yuv(src)

    global COLOR_REFERENCES
    COLOR_REFERENCES = transform_rgb2yuv(COLOR_REFERENCES) if yuv_flag else COLOR_REFERENCES

    if ccm_number in range(1, 4):
        img = colour_correction(
            img_ori, src, COLOR_REFERENCES, method=correction_method_dict[ccm_number], weights=weights
        )
    else:
        raise ValueError('Color correction methods must be in a closed interval [1, 3].')
    
    if yuv_flag:
        img = np.concatenate((np.expand_dims(img[:, :, 0], axis=2), img_ori[:, :, 1:]), axis=2)
        img = cv2.cvtColor(np.clip(img, 0, 255).astype('uint8'), cv2.COLOR_YUV2RGB)
    else:
        img = np.clip(img, 0, 255).astype('uint8')

    return img
