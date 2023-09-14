from pyModbusTCP.client import ModbusClient


client = ModbusClient(
    host='192.168.12.181',
    port=502, auto_open=True,
    debug=False)


def receive_PLC_data(r=20300, r_len=18):
    data_d = {}
    regs = client.read_holding_registers(r, r_len)

    data_d['FTM_sens'] = regs[0]
    data_d['PRF_sens'] = regs[1]
    data_d['tot_holes'] = regs[2]
    data_d['max_bad_holes'] = regs[3]
    data_d['FTM_bias'] = regs[4]
    data_d['FTM_bias_lim'] = regs[5]
    data_d['PRF_bias'] = regs[6]
    data_d['PRF_bias_lim'] = regs[7]
    data_d['aX_coord'] = regs[8]
    data_d['aY_coord'] = regs[9]
    data_d['bX_coord'] = regs[10]
    data_d['bY_coord'] = regs[11]
    data_d['cX_coord'] = regs[12]
    data_d['cY_coord'] = regs[13]
    data_d['dX_coord'] = regs[14]
    data_d['dY_coord'] = regs[15]
    data_d['FTM_corr_interval'] = regs[16]
    data_d['PRF_corr_interval'] = regs[17]

    return data_d


def get_work():
    return client.read_coils(300, 1)[0]


def get_led_brightness():
    return client.read_holding_registers(20318, 2)


def reset_work_flag():
    client.write_single_coil(300, False)


def cam_is_working(status):
    client.write_single_coil(308, status)


def is_DEFECT(def_val):
    client.write_single_coil(302, def_val)
    client.write_single_coil(317, def_val)


def send_DEFF_reason(reasons):
    client.write_multiple_coils(309, reasons)


def rotate_FTM_mot(is_need_rotate, rot_dir):
    client.write_multiple_coils(303, [is_need_rotate, rot_dir])
    client.write_multiple_coils(313, [is_need_rotate, rot_dir])


def rotate_PRF_mot(is_need_rotate, rot_dir):
    client.write_multiple_coils(305, [is_need_rotate, rot_dir])
    client.write_multiple_coils(315, [is_need_rotate, rot_dir])


def send_scan_result(holes, FTM, PRF):
    client.write_multiple_registers(700, [holes, abs(FTM), abs(PRF)])


def send_FTM_bias(avg_FTM, diff_FTM):
    client.write_multiple_registers(
        703,
        [abs(avg_FTM), abs(diff_FTM)])


def send_PRF_bias(avg_PRF, diff_PRF):
    client.write_multiple_registers(
        705,
        [abs(avg_PRF), abs(diff_PRF)])


def send_counters(total, deffects):
    client.write_multiple_registers(
        707, [total, deffects])


def err_too_many_DEFFECTS(err_val):
    client.write_single_coil(312, err_val)


def ERR_raise_internal_error():
    client.write_single_coil(307, True)


def reset_internal_error():
    client.write_single_coil(307, False)
