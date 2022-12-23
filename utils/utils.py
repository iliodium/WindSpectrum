import numpy as np


def get_cmz(model_name, pr_coeff, coordinates):
    cmz = []
    breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
    v2 = breadth
    v3 = breadth + depth
    v4 = 2 * breadth + depth
    mid13_x = breadth / 2
    mid24_x = depth / 2
    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    x, z = coordinates
    x = np.reshape(x, (count_row, -1))
    x = np.split(x, [count_sensors_on_middle,
                     count_sensors_on_middle + count_sensors_on_side,
                     2 * count_sensors_on_middle + count_sensors_on_side,
                     2 * (count_sensors_on_middle + count_sensors_on_side)
                     ], axis=1)

    del x[4]
    x[1] -= v2
    x[2] -= v3
    x[3] -= v4
    mx = np.array([
        abs(x[0] - mid13_x),
        abs(x[1] - mid24_x),
        abs(x[2] - mid13_x),
        abs(x[3] - mid24_x),
    ])
    coeffs_norm_13 = [1 if i <= count_sensors_on_middle // 2 else -1 for i in range(count_sensors_on_middle)]
    coeffs_norm_24 = [1 if i <= count_sensors_on_side // 2 else -1 for i in range(count_sensors_on_side)]
    for coeff in pr_coeff:

        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)
        del coeff[4]
        for i in range(4):
            if i in [0, 2]:
                coeff[i] *= coeffs_norm_13
            else:
                coeff[i] *= coeffs_norm_24
        t_cmz = mx * coeff
        cmz = np.append(cmz, np.sum(t_cmz))
    return np.array(cmz)


def get_cx_cy(model_name, pr_coeff):
    cx = []
    cy = []
    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
    for coeff in pr_coeff:
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)
        del coeff[4]
        faces_x = []
        faces_y = []
        for face in range(len(coeff)):
            if face in [0, 2]:
                faces_x.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))
            else:
                faces_y.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))
        cx.append((faces_x[0] - faces_x[1]))
        cy.append((faces_y[0] - faces_y[1]))
    return np.array(cx), np.array(cy)


if __name__ == '__main__':
    pass
