# RGB to Lab conversion

# Step 1: RGB to XYZ
#         http://www.easyrgb.com/index.php?X=MATH&H=02#text2
# Step 2: XYZ to Lab
#         http://www.easyrgb.com/index.php?X=MATH&H=07#text7
import torch
import numpy as np

def rgb2lab_np(inputColors):
    # inputColors = np.array, [n,3]

    value = inputColors/255
    filter = value > 0.04045
    value[filter] = ((value[filter] + 0.055) / 1.055) ** 2.4
    value[~filter] = value[~filter]/12.92
    RGB = value*100

    XYZ = np.zeros_like(RGB)
    XYZ[:,0] = RGB[:,0] * 0.4124 + RGB[:,1] * 0.3576 + RGB[:,2] * 0.1805
    XYZ[:,1] = RGB[:,0] * 0.2126 + RGB[:,1] * 0.7152 + RGB[:,2] * 0.0722
    XYZ[:,2] = RGB[:,0] * 0.0193 + RGB[:,1] * 0.1192 + RGB[:,2] * 0.9505

    # Observer= 2°, Illuminant= D65
    XYZ[:,0] = XYZ[:,0] / 95.047         # ref_X =  95.047
    XYZ[:,1] = XYZ[:,1] / 100.0          # ref_Y = 100.000
    XYZ[:,2] = XYZ[:,2] / 108.883        # ref_Z = 108.883

    filter = None
    filter = XYZ > 0.008856

    XYZ[filter] = XYZ[filter] ** 0.3333333333333333
    XYZ[~filter] = (7.787 * XYZ[~filter]) + (16 / 116)

    Lab = np.zeros_like(XYZ)
    Lab[:,0] = (116 * XYZ[:,1]) - 16.
    Lab[:,1] = 500 * (XYZ[:,0] - XYZ[:,1])
    Lab[:,2] = 200 * (XYZ[:,1] - XYZ[:,2])

    return Lab

def rgb2lab(inputColor):

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor:
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        RGB[num] = value * 100
        num = num + 1

    XYZ = [0, 0, 0, ]

    X = RGB[0] * 0.4124 + RGB[1] * 0.3576 + RGB[2] * 0.1805
    Y = RGB[0] * 0.2126 + RGB[1] * 0.7152 + RGB[2] * 0.0722
    Z = RGB[0] * 0.0193 + RGB[1] * 0.1192 + RGB[2] * 0.9505
    XYZ[0] = round(X, 4)
    XYZ[1] = round(Y, 4)
    XYZ[2] = round(Z, 4)

    # Observer= 2°, Illuminant= D65
    XYZ[0] = float(XYZ[0]) / 95.047         # ref_X =  95.047
    XYZ[1] = float(XYZ[1]) / 100.0          # ref_Y = 100.000
    XYZ[2] = float(XYZ[2]) / 108.883        # ref_Z = 108.883

    num = 0
    for value in XYZ:

        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)

        XYZ[num] = value
        num = num + 1

    Lab = [0, 0, 0]

    L = (116 * XYZ[1]) - 16
    a = 500 * (XYZ[0] - XYZ[1])
    b = 200 * (XYZ[1] - XYZ[2])

    Lab[0] = round(L, 4)
    Lab[1] = round(a, 4)
    Lab[2] = round(b, 4)

    return Lab

if __name__ == "__main__":
    rgb1 = [12,123,251]
    rgb2 = [3, 123, 112]
    
    testrgbs = np.array([rgb1, rgb2])
    print(f"modified : {rgb2lab_np(testrgbs)}")
    print(f"original : {rgb2lab(rgb1)}\n{rgb2lab(rgb2)}")