W0 = 230 * 1.4


def speed_sp_a(z):
    scale_ks = 1 / 400
    a = 0.15
    k10 = 1
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a


def speed_sp_b(z, scale_ks=1 / 400):
    a = 0.2
    k10 = 0.65
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a


def speed_sp_c(z):
    scale_ks = 1 / 400
    a = 0.25
    k10 = 0.4
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a


def speed_sp_a_m(z):
    scale_ks = 1
    a = 0.15
    k10 = 1
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a


def speed_sp_b_m(z, scale_ks=1):
    a = 0.2
    k10 = 0.65
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a


def speed_sp_c_m(z):
    scale_ks = 1
    a = 0.25
    k10 = 0.4
    po = 1.225
    y = 1
    u10 = (2 * y * k10 * W0 / po) ** 0.5
    return u10 * (scale_ks * z / 10) ** a