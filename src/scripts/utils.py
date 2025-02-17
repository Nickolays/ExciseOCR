import numpy as np
import cv2, os

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def load_image(path:str):
    """ Get RGB image"""
    assert isinstance(path, str)
    assert os.path.exists(path)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess(predict):
    """ 
     Good Combinations:
       - 1 array, with number and series together
       - 2 arrays, with numbers separatle
       - 3 or more arrays with ...

       Concat values, when they have nearest coordinates
       """
    values = []
    probs = []
    for bboxs, val, prob in predict:
        probs.append(prob)
        try:
            num = int(val)   # NUM
            values.append(num)
        except:
            try:
                num = None
                out = val.split(" ")  # Series or "series1 series2 num"
                print(out)
                if len(out) == 2:
                    series = np.array(out[1], dtype=int)    # SERIES
                elif len(out) == 3:
                    num, series = np.array(out[1:], dtype=int)
                    print(num, series)
                else:
                    print("UNKNOWN")
            except:
                print("ERROR")

    return values, probs


# def kmeans_postprocess(model, )

def segmenting_image(model, image):
    """ Use KMeans for 
    image shape is (80, 400, 3)
    """
    assert image.shape[-1] == 3, "Check channel size"

    init_shape = image.shape  # (80, 400, 3)
    # print("   ", init_shape)
    x = image.reshape((-1, 3))
    labels = model.predict(x)
    # print(f"LABELS SHAPE IS {labels.shape}, LENGTH OF IMAGE IS {x.shape}")
    
    for i in range(len(model.cluster_centers_)):
        indxs = np.where(labels == i)
        x[indxs] = model.cluster_centers_[i]

    x = x.reshape(init_shape)
    print(x.shape)

    return image


# roi
def transform_image(hsv_image, mask):
  output = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
  output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
  # output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)

  plt.imshow(output)
  
  return output

def get_y_start_indx(hsv_image):
  # 
  number_mask = cv2.inRange(np.float32(hsv_image),
      lowerb=np.array([170, 50, 0]),
      upperb=np.array([180, 360, 360])
  )

  result = transform_image(hsv_image, number_mask)   # out size is (160, 400)
  # Get min of y index
  result = np.argmin(result[:, 100:, :])
  print(f"RESULT OF INDXS {result}")
  return result

def prepare_image(image):
    """ 
      Obtain ROI of numbers and then crop it

    image - 160, 400
    """
    # Get hsv type for crop
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    y_min_indx = get_y_start_indx(hsv)   # 
    y_min_indx = y_min_indx + 5
    # Crop image
    image = image[y_min_indx:, :, :]
    print(image.shape)
    return image