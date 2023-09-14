import hmisender
import plcsender
import cv2
from math import dist
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from time import perf_counter
import numpy as np
import board
import neopixel


def get_frame():
    # генератор кадров
    for frame in camera.capture_continuous(rawCapture,
                                           format="bgr",
                                           use_video_port=True):
        image = frame.array
        rawCapture.truncate(0)
        yield image


def get_image():
    # первичные преобразования картинки с камеры
    pic = next(get_frame())
    pic = crop_image(pic)
    pic = cv2.rotate(pic, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return pic


def roll_list(lst, elem):
    # FIFO для отбраковки
    lst.insert(0, elem)
    lst.pop()


def led_color_wipe(strip, color, num, wait_ms=10):
    # плавная смена цветов подсветки
    for i in range(num):
        strip[i] = (color)
        time.sleep(wait_ms / 1000.0)


def led_fill(leds, brightness):
    # заливка подсветки с нужной ярокстью для верха и низа
    uppper_br, side_br = brightness

    for led in range(0, 40):
        leds[led] = (uppper_br, uppper_br, uppper_br)
    for led in range(40, 56):
        leds[led] = (side_br, side_br, side_br)


def crop_image(image, x=20, y=220, h=300, w=1000):
    return image[y:y + h, x:x + w]


def calc_midpoint(ptA, ptB):
    # нахождение середины между двумя точками
    ptCx = int((ptA[0] + ptB[0]) * 0.5)
    ptCy = int((ptA[1] + ptB[1]) * 0.5)
    return ptCx, ptCy


def find_PRF_cnts(img, x1, y1, x2, y2, h1):
    # поиск контуров отверстий перфорации

    # рисуем макску в окне поиска отверстий
    mask = np.zeros(img.shape[:2], dtype="uint8")
    x, y = (x1, y1), (x2, y2)
    cv2.rectangle(mask, x, y, 255, -1)

    # накладываем маску
    mskd = cv2.bitwise_and(img, img, mask=mask)

    # конвертируем в bin для поиска контуров
    gray = cv2.cvtColor(mskd, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh_cnt = cv2.threshold(blurred, h1, 255, cv2.THRESH_BINARY)[1]

    # ищем контуры
    cnts = cv2.findContours(thresh_cnt, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    # по площади отбрасываем помехи и слишком мелкие контуры
    good_cnts = []
    for c in cnts:
        epsilon = 0.05 * cv2.arcLength(c, True)
        approx_c = cv2.approxPolyDP(c, epsilon, True)
        if int(cv2.contourArea(approx_c)) >= 15:
            good_cnts.append(approx_c)

    return good_cnts


def find_FTM_dist(img, x1, y1, x2, y2, h1):
    # ищем фотометку и определяем её смещение

    # рисуем маску в зоне поиска
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mskd = cv2.bitwise_and(img, img, mask=mask)

    # конвертируем в bin для поиска контуров
    gray_dist = cv2.cvtColor(mskd, cv2.COLOR_BGR2GRAY)
    blurred_dist = cv2.GaussianBlur(gray_dist, (7, 7), 0)
    thresh_dist = cv2.threshold(
        blurred_dist,
        h1, 255,
        cv2.THRESH_BINARY)[1]

    # ищем контуры
    cnts_dist = cv2.findContours(
        thresh_dist,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)[0]

    # по площади отбрасываем мелкие и большие контуры
    good_c = []
    for c in cnts_dist:
        epsilon = 0.05 * cv2.arcLength(c, True)
        approx_c = cv2.approxPolyDP(c, epsilon, True)
        c_area = int(cv2.contourArea(approx_c))

        if c_area in range(100, 700):
            good_c.append(c)

    # должен быть только один подходящий контур, иначе брак
    if len(good_c) == 1:

        # вписываем прямоугольник в найденный контур
        rect = cv2.minAreaRect(good_c[0])
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")

        # находим центр прямоугольника
        M = cv2.moments(box)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        FTM_coords = [[cX, cY]]

        # устанавливаем точку остчёта
        # X - координата центра фотометки
        # Y - нижний край картинки
        refCoords = [(cX, img.shape[0])]

        # ищем расстояние от края кадра до центра
        for ((xA, yA), (xB, yB)) in zip(refCoords, FTM_coords):
            D = int(dist((xA, yA), (xB, yB)) / 8)

            # рисуем линию и указываем в её середине текст
            cv2.line(image, (int(xA), int(yA)), (int(xB), int(yB)),
                     (0, 255, 0), 2)
            txtXcoord, txtYcoord = calc_midpoint((xA, yA), (xB, yB))
            cv2.putText(
                image, '{}'.format(D),
                (txtXcoord - 40, txtYcoord),
                font, 0.75, (0, 255, 127), 1)

        # рисуем найденный контур и точки отсчёта
        cv2.drawContours(image, [box], -1, (0, 255, 0), 1)
        cv2.circle(image, (cX, cY), 3, (255, 0, 255), -1)
        cv2.circle(image, refCoords[0], 5, (0, 0, 255), -1)

        return D, FTM_coords

    else:
        return 0, [[0, 0]]


def find_PRF_dist(img, PRF_cnts, FTM_xy):
    # ищем смещение первого ряда перфорации от фотометки
    sum_cX = 0
    sum_cY = 0

    FTMx, FTMy = FTM_xy[0]

    # заливаем три первых выбранных контура
    cv2.drawContours(img, PRF_cnts[0:3], -1, (255, 0, 0), cv2.FILLED)

    # находим центры выбранных контуров
    for i, c in enumerate(PRF_cnts[0:3]):
        Mom = cv2.moments(c)
        cX = int(Mom["m10"] / Mom["m00"])
        cY = int(Mom["m01"] / Mom["m00"])
        sum_cX += cX
        sum_cY += cY

    # считаем средние координаты
    avg_cY = int(sum_cY / 3)

    # считаем расстояние смещения
    d = int(dist((FTMx, FTMy), (FTMx, avg_cY)) / 8)

    # меняем знак
    if avg_cY > FTMy:
        d = d * -1

    return d, avg_cY


def check_result_is_OK(finded, ok, max_bad):

    if finded in range(ok - max_bad, ok + max_bad):
        return True
    else:
        return False


def open_rejector(def_val):
    global reject_list

    roll_list(reject_list, def_val)

    to_set = set(reject_list)

    # если все результаты подряд брак, то ошибка с аварией
    if len(to_set) == 1 and 1 in to_set:
        plcsender.is_DEFECT(True)
        plcsender.err_too_many_DEFFECTS(True)
        print('ERR - too many deffects')

    elif reject_list[-1] == 1:
        plcsender.is_DEFECT(True)
        plcsender.err_too_many_DEFFECTS(False)
        print('DEFECT, open rejector')

    else:
        plcsender.is_DEFECT(False)
        plcsender.err_too_many_DEFFECTS(False)


def start_FTM_adj(f_FTM, F_bias):

    avg_FTM = int(np.median(f_FTM))
    FTM_diff = F_bias - avg_FTM
    plcsender.send_FTM_bias(avg_FTM, FTM_diff)

    if FTM_diff == 0:
        plcsender.rotate_FTM_mot(False, False)
    elif FTM_diff > 0:
        plcsender.rotate_FTM_mot(True, True)
        print('rotating FTM CCW...')
    else:
        plcsender.rotate_FTM_mot(True, False)
        print('rotating FTM CW...')


def start_PRF_adj(f_PRF, P_bias):
    avg_PRF = int(np.median(f_PRF))
    PRF_diff = P_bias - avg_PRF
    plcsender.send_PRF_bias(avg_PRF, PRF_diff)

    if PRF_diff == 0:
        plcsender.rotate_PRF_mot(False, False)
    elif PRF_diff > 0:
        plcsender.rotate_PRF_mot(True, True)
        print('rotating PRF CCW...')
    else:
        plcsender.rotate_PRF_mot(True, False)
        print('rotating PRF CW...')


print('start init')

led_num = 56
LEDS = neopixel.NeoPixel(board.D10, led_num)
led_color_wipe(LEDS, (255, 63, 7), led_num)
led_color_wipe(LEDS, (7, 147, 223), led_num)
led_color_wipe(LEDS, (255, 63, 7), led_num)
LEDS.fill((0, 0, 0))
print('leds init done...')


# init camera
cam_res = (1024, 768)
camera = PiCamera()
camera.resolution = cam_res
camera.iso = 100
rawCapture = PiRGBArray(camera, size=cam_res)
time.sleep(2)
next(get_frame())
print('camera init done...')


# ожидание подключения к ПЛК
while plcsender.client.open() is False:
    print('modbus TCP connection error...')
    led_color_wipe(LEDS, (200, 0, 0), led_num)
    print('trying to reconnect')
    led_color_wipe(LEDS, (0, 0, 0), led_num)

print('modbusTCP connection is open...')

led_fill(LEDS, plcsender.get_led_brightness())

# сброс флага ошибки ПО камеры
# делается на тот случай, если камера аварийно завершила работу
plcsender.reset_internal_error()

font = cv2.FONT_HERSHEY_SIMPLEX
reject_list = [0] * 7
global_deff_counter = 0
global_scan_counter = 0
FTM_adjust_counter = 0
PRF_adjust_counter = 0
FTM_dist_accum = []
PRF_dist_accum = []

print('init done')

try:
    while True:
        if plcsender.get_work() is True:

            tmr_start = perf_counter()

            # сбрасываем флаг старта работы
            plcsender.reset_work_flag()

            # вывешиваем флаг начала работы
            plcsender.cam_is_working(True)

            # устанавливаем яркость подсветки
            led_fill(LEDS, plcsender.get_led_brightness())

            # получаем данные для работы
            PLC_data = plcsender.receive_PLC_data()

            image = get_image()
            # cv2.imwrite('111.jpg', image)  # debug write

            # обрабатываем картинку
            # поиск отверстий перфорации
            finded_PRF_conts = find_PRF_cnts(
                image,
                PLC_data['aX_coord'],
                PLC_data['aY_coord'],
                PLC_data['bX_coord'],
                PLC_data['bY_coord'],
                PLC_data['PRF_sens'])

            # поиск фотометки и её смещения
            finded_FTM_dist, finded_FTM_coords = find_FTM_dist(
                image,
                PLC_data['cX_coord'],
                PLC_data['cY_coord'],
                PLC_data['dX_coord'],
                PLC_data['dY_coord'],
                PLC_data['FTM_sens'])

            # поиск первого ряда перфорации и его смещения от фотометки
            finded_PRF_dist, p_avg_cY = find_PRF_dist(
                image,
                finded_PRF_conts,
                finded_FTM_coords)

            # распаковываем координаты центра фотометки
            label_coords_X, label_coords_Y = finded_FTM_coords[0]

            # рисование наглядного отобажения координат
            cv2.arrowedLine(image, (10, 10), (10, 70), (255, 255, 255), 1)
            cv2.arrowedLine(image, (10, 10), (70, 10), (255, 255, 255), 1)
            cv2.putText(image, "y", (15, 65), font, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "x", (75, 15), font, 0.5, (255, 255, 255), 1)

            # рисование контуров отверстий
            cv2.drawContours(image,
                             finded_PRF_conts,
                             -1, (7, 63, 255),
                             cv2.FILLED)

            # рисование РОИ нахождения контуров отверстий
            cv2.rectangle(
                image,
                (PLC_data['aX_coord'], PLC_data['aY_coord']),
                (PLC_data['bX_coord'], PLC_data['bY_coord']),
                (255, 255, 255), 1)

            # рисование РОИ нахождения контуров фотометки
            cv2.rectangle(
                image,
                (PLC_data['cX_coord'], PLC_data['cY_coord']),
                (PLC_data['dX_coord'], PLC_data['dY_coord']),
                (0, 255, 255), 1)

            # рисование верхней и нижней точки РОИ отверстий
            cv2.circle(
                image,
                (PLC_data['aX_coord'], PLC_data['aY_coord']),
                4, (0, 0, 255), -1)
            cv2.circle(
                image,
                (PLC_data['bX_coord'], PLC_data['bY_coord']),
                4, (0, 0, 255), -1)

            cv2.putText(
                image, "A",
                ((PLC_data['aX_coord'] + 5),
                 (PLC_data['aY_coord'] - 5)),
                font, 1, (255, 255, 255), 2)

            cv2.putText(
                image, "B",
                ((PLC_data['bX_coord'] - 25),
                 (PLC_data['bY_coord'] + 25)),
                font, 1, (255, 255, 255), 2)

            # рисование верхней и нижней точки РОИ фотометки
            cv2.circle(
                image,
                (PLC_data['cX_coord'], PLC_data['cY_coord']),
                4, (0, 0, 255), -1)
            cv2.circle(
                image,
                (PLC_data['dX_coord'], PLC_data['dY_coord']),
                4, (0, 0, 255), -1)

            cv2.putText(
                image, "C",
                ((PLC_data['cX_coord'] + 5),
                 (PLC_data['cY_coord'] - 5)),
                font, 1, (255, 255, 255), 2)

            cv2.putText(
                image, "D",
                ((PLC_data['dX_coord'] - 25),
                 (PLC_data['dY_coord'] + 25)),
                font, 1, (255, 255, 255), 2)

            # рисование расстояния до перфорации
            cv2.putText(
                image,
                '{}'.format(finded_PRF_dist),
                (label_coords_X - 30, label_coords_Y + 25),
                font, 0.75, (0, 0, 255), 1)

            # рисование линии расстояния до перфорации
            cv2.line(
                image,
                (label_coords_X, label_coords_Y),
                (label_coords_X, p_avg_cY),
                (60, 0, 255), 2)

            # рисование линии первого ряда перфорации
            cv2.line(
                image,
                (20, p_avg_cY),
                (label_coords_X, p_avg_cY),
                (255, 127, 0), 1)

            # собираем результаты сканирования
            scan_results = []
            scan_results.append(
                check_result_is_OK(len(finded_PRF_conts),
                                   PLC_data['tot_holes'],
                                   PLC_data['max_bad_holes']))
            scan_results.append(
                check_result_is_OK(finded_FTM_dist,
                                   PLC_data['FTM_bias'],
                                   PLC_data['FTM_bias_lim']))
            scan_results.append(
                check_result_is_OK(finded_PRF_dist,
                                   PLC_data['PRF_bias'],
                                   PLC_data['PRF_bias_lim']))

            # проверка результатов
            if False in scan_results:
                open_rejector(1)  # брак
                global_deff_counter += 1  # увеличиваем счётчик брака
            else:
                open_rejector(0)  # не брак

            # отправка счётчиков в ПЛК
            plcsender.send_counters(
                global_scan_counter,
                global_deff_counter)

            # отправка результатов в ПЛК
            plcsender.send_scan_result(
                len(finded_PRF_conts),
                finded_FTM_dist,
                finded_PRF_dist)
            plcsender.send_DEFF_reason(scan_results)

            # копим значения для коррекции
            FTM_dist_accum.append(finded_FTM_dist)
            PRF_dist_accum.append(finded_PRF_dist)

            # увеличиваем счётчики
            global_scan_counter += 1
            FTM_adjust_counter += 1
            PRF_adjust_counter += 1

            # проверка необходимиости коррекции фотометки
            if FTM_adjust_counter >= PLC_data['FTM_corr_interval']:
                # начинаем двигать мотором
                start_FTM_adj(
                    FTM_dist_accum,
                    PLC_data['FTM_bias'],)
                # обнуляем данные для коррекции
                FTM_adjust_counter = 0
                FTM_dist_accum = []
            else:
                plcsender.rotate_FTM_mot(False, False)

            # проверка необходимиости коррекции перфорации
            if PRF_adjust_counter >= PLC_data['PRF_corr_interval']:
                # начинаем двигать мотором
                start_PRF_adj(
                    PRF_dist_accum,
                    PLC_data['PRF_bias'],)
                # обнуляем данные для коррекции
                PRF_adjust_counter = 0
                PRF_dist_accum = []
            else:
                plcsender.rotate_PRF_mot(False, False)

            # отправка итоговой графики на тач-панель
            hmisender.send_img_to_HMI(image)

            # сигнал для плк обработка закончена
            plcsender.cam_is_working(False)

            tmr_end = perf_counter()
            t_diff = tmr_end - tmr_start
            print(
                f'tmr={t_diff:.4f}s, fr={global_scan_counter}, {scan_results}')

        else:
            time.sleep(0.1)

finally:
    LEDS.fill((0, 0, 0))
    print('\nUnexpected error... \nCleanup!')
    plcsender.cam_is_working(False)

    # активируем флаг внутренней ошибки ПО камеры
    plcsender.ERR_raise_internal_error()

    camera.close()
