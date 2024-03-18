def mk_xy2pix():
    x2pix = [0] * 128
    y2pix = [0] * 128

    for I in range(1, 129):
        J = I - 1
        K = 0
        IP = 1
        while J != 0:
            ID = J % 2
            J = J // 2
            K = IP * ID + K
            IP = IP * 4
        
        x2pix[I-1] = K
        y2pix[I-1] = 2 * K

    return x2pix, y2pix

print(mk_xy2pix())
