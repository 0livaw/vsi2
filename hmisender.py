import cv2
from fastcrc import crc32
import socket
import numpy as np
import textwrap


def img_to_hex(img):
    # decoding to 565 color
    im555 = np.vstack(cv2.cvtColor(img, cv2.COLOR_BGR2BGR565))
    img_bytes = im555.tobytes()
    return img_bytes.hex()


def calc_crc32(bytes_to_calc):
    _crc32 = hex(crc32.iso_hdlc(bytes_to_calc))[2:].rjust(8, '0')
    crc32_rev = _crc32[-2:] + _crc32[-4:-2] + _crc32[-6:-4] + _crc32[-8:-6]
    return crc32_rev


def split_data_to_packets(data, tot=16384):

    list_of_packets = textwrap.wrap(data, tot)
    for ind, pack in enumerate(list_of_packets):
        list_of_packets[ind] = calc_crc32(
            bytes.fromhex(list_of_packets[ind])) + list_of_packets[ind]

    return list_of_packets


def send_img_to_HMI(cv2_image):
    HOST = '192.168.12.180'
    PORT = 12347

    image_hex_header = '012c03e81000000000000000'
    first_com = '434d440014000000010000000200000000000000'
    second_com = '434d4400140000000300000002000000cc270900'

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        s.sendall(bytes.fromhex(first_com))
        data = s.recv(1024)

        if len(data) > 0:

            image_hex = image_hex_header + img_to_hex(cv2_image)
            data_to_send = split_data_to_packets(image_hex)

            s.sendall(bytes.fromhex(second_com))
            data = s.recv(1024)

            for i, elem in enumerate(data_to_send):
                s.sendall(i.to_bytes(2, 'little'))
                try:
                    s.sendall(bytes.fromhex(elem))
                except ValueError:
                    print('ValueError')
                    return False

                if i == 0 and len(data_to_send) > 1:
                    data = s.recv(1024)
                data = s.recv(1024)

                if 'CMD_OK' in str(data):
                    continue
                else:
                    return False
            return True
        else:
            return None
