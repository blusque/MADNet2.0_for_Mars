import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import time

pi = 3.1415926
half_round = 180


class HillShadeParam:

    def __init__(self, z_factor, altitude=45.0, azimuth=315.0):
        azimuth = azimuth if azimuth < 360 else azimuth - 360
        self.z_factor = z_factor
        self.zenith_rad = (90 - altitude) * pi / half_round
        self.azimuth_rad = (360 - azimuth + 90) * pi / half_round


def hill_shade_alg(dx, dy, param: HillShadeParam):
    slope_rad = np.arctan(param.z_factor * np.sqrt(dx ** 2 + dy ** 2))
    aspect_rad = np.arctan2(dy, -dx)

    row = aspect_rad.shape[0]
    col = aspect_rad.shape[1]

    start = time.time()
    for i in range(row):
        for j in range(col):
            if aspect_rad[i, j] < 0:
                aspect_rad[i, j] = 2 * pi + aspect_rad[i, j]
            elif aspect_rad[i, j] == 0:
                if dy[i, j] > 0:
                    aspect_rad[i, j] = pi / 2
                elif dy[i, j] < 0:
                    aspect_rad[i, j] = 2 * pi - pi / 2
    end = time.time()
    print("loop cost: ", end - start)

    azimuth_rad_matrix = np.array([[param.azimuth_rad for i in range(col)] for j in range(row)], dtype=np.float32)
    shade = (np.cos(param.zenith_rad) * np.cos(slope_rad)) \
            + (np.sin(param.zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad_matrix - aspect_rad))

    return shade


def hill_shade(input_data, z_factor, altitude=45.0, azimuth=315.0):
    shade_param = HillShadeParam(z_factor, altitude, azimuth)

    dx, dy = np.gradient(input_data)

    relief = hill_shade_alg(dx, dy, shade_param)

    return relief


def main():
    filename = '/media/mei/Elements/dem_image/dtm/001462_2015-DTM_12.tif'
    dtm = io.imread(filename, plugin='pil')
    io.imshow(dtm)
    plt.show()
    start = time.time()
    relief = hill_shade(dtm, z_factor=1.0)
    end = time.time()
    print("relief cost: ", end - start)
    io.imshow(relief)
    plt.show()


if __name__ == "__main__":
    main()

