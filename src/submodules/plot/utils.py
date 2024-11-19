import numpy as np


def get_labels(point):
    """
    эта функция нужна тк
    np.arange(0, 0.3, 0.1) -> array([0. , 0.1, 0.2])

    np.arange(0, 0.2 + 0.1, 0.1) -> array([0. , 0.1, 0.2, 0.3])
    """
    point = int(point * 10)
    arange = np.arange(0, point + 1)

    return [f"{i / 10:.1f}" for i in arange]


def scaling_data(x, y=None, angle_border=50):
    """Масштабирование данных до 360 градусов для графиков в полярной системе координат для изолированных зданий"""
    match angle_border:
        case 50:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                c = np.append(b, np.flip(b)[1:])
                y_scale = np.append(c, np.flip(c)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])
                return x_scale

        case 95:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                y_scale = np.append(b, np.flip(b)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])
                return x_scale

def round_list_model_pic(arr):
    new_arr = []
    for val in arr:
        if isinstance(val, int):
            new_arr.append(str(val))
            continue

        elif isinstance(val, str):
            val = float(val)

        new_arr.append(f'{round(val, 2):.1f}')

    return new_arr

if __name__ == "__main__":
    print(get_labels(0.5))
