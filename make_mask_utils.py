import np
import matplotlib.pyplot as plt
from skimage import measure



def show_mask(mask, ax, random_color=False):
    """
    mask 시각화 함수
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    return mask_image
    
    
def show_points(coords, labels, ax, marker_size=375):
    """
    point 시각화 함수
    """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_box(box, ax):
    """
    bbox 시각화 함수
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



def remove_small_components(mask, min_size=1000):
    """
    mask: sam model에서 inference 된 마스크
    
    return: min_size보다 작은 구성 요소가 제거된 마스크
    """
    labeled_mask = measure.label(mask)
    unique_labels = np.unique(labeled_mask)
    
    for label in unique_labels:
        if np.sum(labeled_mask == label) < min_size:
            mask[labeled_mask == label] = 0

    return mask