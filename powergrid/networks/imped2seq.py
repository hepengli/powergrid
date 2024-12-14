import numpy as np

def imped2seq(Z_abc):
    a = np.exp(1j*np.deg2rad(120))
    A = np.array([
        [1., 1., 1.],
        [1., a**2, a],
        [1., a, a**2]
    ])
    return np.linalg.inv(A).dot(Z_abc).dot(A)

if __name__ == '__main__':
    np.set_printoptions(precision=7)
    # Config 300
    Z_abc = np.array([
        [1.3368+1j*1.3343, 0.2101+1j*0.5779, 0.2130+1j*0.5015],
        [0.2101+1j*0.5779, 1.3238+1j*1.3569, 0.2066+1j*0.4591],
        [0.2130+1j*0.5015, 0.2066+1j*0.4591, 1.3294+1j*1.3471]
    ]) # ohms/mile
    B_abc = np.array([
        [5.3350,  -1.5313, -0.9943],
        [-1.5313,  5.0979, -0.6212],
        [-0.9943, -0.6212,  4.8880],
    ]) # micro Siemens /mile

    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 1
    Z_abc = np.array([
        [0.4576+1j*1.0780, 0.1560+1j*0.5017, 0.1535+1j*0.3849],
        [0.1560+1j*0.5017, 0.4666+1j*1.0482, 0.1580+1j*0.4236],
        [0.1535+1j*0.3849, 0.1580+1j*0.4236, 0.4615+1J*1.0651]
    ])
    B_abc = np.array([
        [5.6765,  -1.8319, -0.6982],
        [-1.8319,  5.9809, -1.1645],
        [-0.6982, -1.1645,  5.3971]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 2
    Z_abc = np.array([
        [0.4666+1j*1.0482, 0.1580+1j*0.4236, 0.1560+1j*0.5017],
        [0.1580+1j*0.4236, 0.4615+1j*1.0651, 0.1535+1j*0.3849],
        [0.1560+1j*0.5017, 0.1535+1j*0.3849, 0.4576+1j*1.0780]
    ])
    B_abc = np.array([
        [5.9809,  -1.1645, -1.8319],
        [-1.1645,  5.3971, -0.6982],
        [-1.8319, -0.6982,  5.6765]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 3
    Z_abc = np.array([
        [0.4615+1j*1.0651, 0.1535+1j*0.3849, 0.1580+1j*0.4236],
        [0.1535+1j*0.3849, 0.4576+1j*1.0780, 0.1560+1j*0.5017],
        [0.1580+1j*0.4236, 0.1560+1j*0.5017, 0.4666+1j*1.0482]
    ])
    B_abc = np.array([
        [5.3971, -0.6982, -1.1645],
        [-0.6982, 5.6765, -1.8319],
        [-1.1645,-1.8319,  5.9809]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 4
    Z_abc = np.array([
        [0.4615+1j*1.0651, 0.1580+1j*0.4236, 0.1535+1j*0.3849],
        [0.1580+1j*0.4236, 0.4666+1j*1.0482, 0.1560+1j*0.5017],
        [0.1535+1j*0.3849, 0.1560+1j*0.5017, 0.4576+1j*1.0780]
    ])
    B_abc = np.array([
        [5.3971, -1.1645, -0.6982],
        [-1.1645, 5.9809, -1.8319],
        [-0.6982,-1.8319,  5.6765]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 5
    Z_abc = np.array([
        [0.4666+1j*1.0482, 0.1560+1j*0.5017, 0.1580+1j*0.4236],
        [0.1560+1j*0.5017, 0.4576+1j*1.0780, 0.1535+1j*0.3849],
        [0.1580+1j*0.4236, 0.1535+1j*0.3849, 0.4615+1j*1.0651]
    ])
    B_abc = np.array([
        [5.9809, -1.8319, -1.1645],
        [-1.8319, 5.6765, -0.6982],
        [-1.1645,-0.6982,  5.3971],
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 6
    Z_abc = np.array([
        [0.4576+1j*1.0780, 0.1535+1j*0.3849, 0.1560+1j*0.5017],
        [0.1535+1j*0.3849, 0.4615+1j*1.0651, 0.1580+1j*0.4236],
        [0.1560+1j*0.5017, 0.1580+1j*0.4236, 0.4666+1j*1.0482]
    ])
    B_abc = np.array([
        [5.6765, -0.6982, -1.8319],
        [-0.6982, 5.3971, -1.1645],
        [-1.8319,-1.1645,  5.9809]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 7
    Z_abc = np.array([
        [0.4576+1j*1.0780, 0.0000+1j*0.0000, 0.1535+1j*0.3849],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.1535+1j*0.3849, 0.0000+1j*0.0000, 0.4615+1j*1.0651]
    ])
    B_abc = np.array([
        [5.1154, 0.0000, -1.0549],
        [0.0000, 0.0000,  0.0000],
        [-1.0549,0.0000,  5.1704],
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 8
    Z_abc = np.array([
        [0.4576+1j*1.0780, 0.1535+1j*0.3849, 0.0000+1j*0.0000],
        [0.1535+1j*0.3849, 0.4615+1j*1.0651, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000]
    ])
    B_abc = np.array([
        [5.1154, -1.0549, 0.0000],
        [-1.0549, 5.1704, 0.0000],
        [0.0000,  0.0000, 0.0000]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 9
    Z_abc = np.array([
        [1.3292+1j*1.3475, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000]
    ])
    B_abc = np.array([
        [4.5193, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 10
    Z_abc = np.array([
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 1.3292+1j*1.3475, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000]
    ])
    B_abc = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 4.5193, 0.0000],
        [0.0000, 0.0000, 0.0000]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 11
    Z_abc = np.array([
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 1.3292+1j*1.3475]
    ])
    B_abc = np.array([
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 4.5193]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))

    # Config 12
    Z_abc = np.array([
        [1.5209+1j*0.7521, 0.5198+1j*0.2775, 0.4924+1j*0.2157],
        [0.5198+1j*0.2775, 1.5329+1j*0.7162, 0.5198+1j*0.2775],
        [0.4924+1j*0.2157, 0.5198+1j*0.2775, 1.5209+1j*0.7521]
    ])
    B_abc = np.array([
        [67.2242, 0.0000, 0.0000],
        [0.0000, 67.2242, 0.0000],
        [0.0000,  0.0000, 67.2242]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 601
    Z_abc = np.array([
        [0.3465+1j*1.0179, 0.1560+1j*0.5017, 0.1580+1j*0.4236],
        [0.1560+1j*0.5017, 0.3375+1j*1.0478, 0.1535+1j*0.3849],
        [0.1580+1j*0.4236, 0.1535+1j*0.3849, 0.3414+1j*1.0348]
    ])
    B_abc = np.array([
        [6.2998 , -1.9958, -1.2595],
        [-1.9958,  5.9597, -0.7417],
        [-1.2595, -0.7417,  5.6386]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 601: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 602
    Z_abc = np.array([
        [0.7526+1j*1.1814, 0.1580+1j*0.4236, 0.1560+1j*0.5017],
        [0.1580+1j*0.4236, 0.7475+1j*1.1983, 0.1535+1j*0.3849],
        [0.1560+1j*0.5017, 0.1535+1j*0.3849, 0.7436+1j*1.2112]
    ])
    B_abc = np.array([
        [5.6990 , -1.0817, -1.6905],
        [-1.0817,  5.1795, -0.6588],
        [-1.6905, -0.6588,  5.4246]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 602: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 603
    Z_abc = np.array([
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 1.3294+1j*1.3471, 0.2066+1j*0.4591],
        [0.0000+1j*0.0000, 0.2066+1j*0.4591, 1.3238+1j*1.3569]
    ])
    B_abc = np.array([
        [0.0000,  0.0000,  0.0000],
        [0.0000,  4.7097, -0.8999],
        [0.0000, -0.8999,  4.6658]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 603: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 604
    Z_abc = np.array([
        [1.3238+1j*1.3569, 0.0000+1j*0.0000, 0.2066+1j*0.4591],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 1.3294+1j*1.3471]
    ])
    B_abc = np.array([
        [0.0000,  0.0000,  0.8999],
        [0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  4.7097]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 604: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 605
    Z_abc = np.array([
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000, 0.0000+1j*0.0000, 1.3294+1j*1.3475]
    ])
    B_abc = np.array([
        [0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  0.0000],
        [0.0000,  0.0000,  4.5193]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 605: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 606
    Z_abc = np.array([
        [0.7982+1j*0.4463,  0.3192+1j*0.0328, 0.2849+1j*-0.0143],
        [0.3192+1j*0.0328,  0.7891+1j*0.4041, 0.3192+1j*0.0328],
        [0.2849+1j*-0.0143, 0.3192+1j*0.0328, 0.7982+1j*0.4463]
    ])
    B_abc = np.array([
        [96.8897,  0.0000,  0.0000],
        [0.0000,  96.8897,  0.0000],
        [0.0000,   0.0000,  96.8897]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 606: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))


    # Config 607
    Z_abc = np.array([
        [1.3425+1j*0.5124,  0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000,  0.0000+1j*0.0000, 0.0000+1j*0.0000],
        [0.0000+1j*0.0000,  0.0000+1j*0.0000, 0.0000+1j*0.0000]
    ])
    B_abc = np.array([
        [88.9912,  0.0000,  0.0000],
        [0.0000,   0.0000,  0.0000],
        [0.0000,   0.0000,  0.0000]
    ])
    Z_012 = imped2seq(Z_abc) / 1.609344 # ohms/km
    B_012 = imped2seq(B_abc) / 1.609344 # micro Siemens/km
    print('\nConfig 607: ')
    print('c1: {}'.format(B_012[1,1].real))
    print('r1: {}'.format(Z_012[1,1].real))
    print('x1: {}'.format(Z_012[1,1].imag))
    print('c0: {}'.format(B_012[0,0].real))
    print('r0: {}'.format(Z_012[0,0].real))
    print('x0: {}'.format(Z_012[0,0].imag))
