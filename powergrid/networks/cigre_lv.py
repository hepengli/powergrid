from numpy import nan
from pandas import read_json
import pandapower as pp

def CIGRE_LV():
    """
    Create the CIGRE LV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net_cigre_lv = pp.create_empty_network()

    # Linedata
    # UG1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.162,
                 'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG1', element='line')

    # UG2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.2647,
                 'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG2', element='line')

    # UG3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.822,
                 'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG3', element='line')

    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH3', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_lv, name='Bus 0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR0 = pp.create_bus(net_cigre_lv, name='Bus R0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR1 = pp.create_bus(net_cigre_lv, name='Bus R1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busR2 = pp.create_bus(net_cigre_lv, name='Bus R2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR3 = pp.create_bus(net_cigre_lv, name='Bus R3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR4 = pp.create_bus(net_cigre_lv, name='Bus R4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR5 = pp.create_bus(net_cigre_lv, name='Bus R5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR6 = pp.create_bus(net_cigre_lv, name='Bus R6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR7 = pp.create_bus(net_cigre_lv, name='Bus R7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR8 = pp.create_bus(net_cigre_lv, name='Bus R8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR9 = pp.create_bus(net_cigre_lv, name='Bus R9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR10 = pp.create_bus(net_cigre_lv, name='Bus R10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR11 = pp.create_bus(net_cigre_lv, name='Bus R11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR12 = pp.create_bus(net_cigre_lv, name='Bus R12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR13 = pp.create_bus(net_cigre_lv, name='Bus R13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR14 = pp.create_bus(net_cigre_lv, name='Bus R14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR15 = pp.create_bus(net_cigre_lv, name='Bus R15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR16 = pp.create_bus(net_cigre_lv, name='Bus R16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR17 = pp.create_bus(net_cigre_lv, name='Bus R17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR18 = pp.create_bus(net_cigre_lv, name='Bus R18', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busI0 = pp.create_bus(net_cigre_lv, name='Bus I0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busI1 = pp.create_bus(net_cigre_lv, name='Bus I1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busI2 = pp.create_bus(net_cigre_lv, name='Bus I2', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busC0 = pp.create_bus(net_cigre_lv, name='Bus C0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busC1 = pp.create_bus(net_cigre_lv, name='Bus C1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busC2 = pp.create_bus(net_cigre_lv, name='Bus C2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC3 = pp.create_bus(net_cigre_lv, name='Bus C3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC4 = pp.create_bus(net_cigre_lv, name='Bus C4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC5 = pp.create_bus(net_cigre_lv, name='Bus C5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC6 = pp.create_bus(net_cigre_lv, name='Bus C6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC7 = pp.create_bus(net_cigre_lv, name='Bus C7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC8 = pp.create_bus(net_cigre_lv, name='Bus C8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC9 = pp.create_bus(net_cigre_lv, name='Bus C9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC10 = pp.create_bus(net_cigre_lv, name='Bus C10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC11 = pp.create_bus(net_cigre_lv, name='Bus C11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC12 = pp.create_bus(net_cigre_lv, name='Bus C12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC13 = pp.create_bus(net_cigre_lv, name='Bus C13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC14 = pp.create_bus(net_cigre_lv, name='Bus C14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC15 = pp.create_bus(net_cigre_lv, name='Bus C15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC16 = pp.create_bus(net_cigre_lv, name='Bus C16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC17 = pp.create_bus(net_cigre_lv, name='Bus C17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC18 = pp.create_bus(net_cigre_lv, name='Bus C18', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC19 = pp.create_bus(net_cigre_lv, name='Bus C19', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC20 = pp.create_bus(net_cigre_lv, name='Bus C20', vn_kv=0.4, type='m', zone='CIGRE_LV')

    # Lines
    pp.create_line(net_cigre_lv, busR1, busR2, length_km=0.035, std_type='UG1',
                   name='Line R1-R2')
    pp.create_line(net_cigre_lv, busR2, busR3, length_km=0.035, std_type='UG1',
                   name='Line R2-R3')
    pp.create_line(net_cigre_lv, busR3, busR4, length_km=0.035, std_type='UG1',
                   name='Line R3-R4')
    pp.create_line(net_cigre_lv, busR4, busR5, length_km=0.035, std_type='UG1',
                   name='Line R4-R5')
    pp.create_line(net_cigre_lv, busR5, busR6, length_km=0.035, std_type='UG1',
                   name='Line R5-R6')
    pp.create_line(net_cigre_lv, busR6, busR7, length_km=0.035, std_type='UG1',
                   name='Line R6-R7')
    pp.create_line(net_cigre_lv, busR7, busR8, length_km=0.035, std_type='UG1',
                   name='Line R7-R8')
    pp.create_line(net_cigre_lv, busR8, busR9, length_km=0.035, std_type='UG1',
                   name='Line R8-R9')
    pp.create_line(net_cigre_lv, busR9, busR10, length_km=0.035, std_type='UG1',
                   name='Line R9-R10')
    pp.create_line(net_cigre_lv, busR3, busR11, length_km=0.030, std_type='UG3',
                   name='Line R3-R11')
    pp.create_line(net_cigre_lv, busR4, busR12, length_km=0.035, std_type='UG3',
                   name='Line R4-R12')
    pp.create_line(net_cigre_lv, busR12, busR13, length_km=0.035, std_type='UG3',
                   name='Line R12-R13')
    pp.create_line(net_cigre_lv, busR13, busR14, length_km=0.035, std_type='UG3',
                   name='Line R13-R14')
    pp.create_line(net_cigre_lv, busR14, busR15, length_km=0.030, std_type='UG3',
                   name='Line R14-R15')
    pp.create_line(net_cigre_lv, busR6, busR16, length_km=0.030, std_type='UG3',
                   name='Line R6-R16')
    pp.create_line(net_cigre_lv, busR9, busR17, length_km=0.030, std_type='UG3',
                   name='Line R9-R17')
    pp.create_line(net_cigre_lv, busR10, busR18, length_km=0.030, std_type='UG3',
                   name='Line R10-R18')

    pp.create_line(net_cigre_lv, busI1, busI2, length_km=0.2, std_type='UG2',
                   name='Line I1-I2')

    pp.create_line(net_cigre_lv, busC1, busC2, length_km=0.030, std_type='OH1',
                   name='Line C1-C2')
    pp.create_line(net_cigre_lv, busC2, busC3, length_km=0.030, std_type='OH1',
                   name='Line C2-C3')
    pp.create_line(net_cigre_lv, busC3, busC4, length_km=0.030, std_type='OH1',
                   name='Line C3-C4')
    pp.create_line(net_cigre_lv, busC4, busC5, length_km=0.030, std_type='OH1',
                   name='Line C4-C5')
    pp.create_line(net_cigre_lv, busC5, busC6, length_km=0.030, std_type='OH1',
                   name='Line C5-C6')
    pp.create_line(net_cigre_lv, busC6, busC7, length_km=0.030, std_type='OH1',
                   name='Line C6-C7')
    pp.create_line(net_cigre_lv, busC7, busC8, length_km=0.030, std_type='OH1',
                   name='Line C7-C8')
    pp.create_line(net_cigre_lv, busC8, busC9, length_km=0.030, std_type='OH1',
                   name='Line C8-C9')
    pp.create_line(net_cigre_lv, busC3, busC10, length_km=0.030, std_type='OH2',
                   name='Line C3-C10')
    pp.create_line(net_cigre_lv, busC10, busC11, length_km=0.030, std_type='OH2',
                   name='Line C10-C11')
    pp.create_line(net_cigre_lv, busC11, busC12, length_km=0.030, std_type='OH3',
                   name='Line C11-C12')
    pp.create_line(net_cigre_lv, busC11, busC13, length_km=0.030, std_type='OH3',
                   name='Line C11-C13')
    pp.create_line(net_cigre_lv, busC10, busC14, length_km=0.030, std_type='OH3',
                   name='Line C10-C14')
    pp.create_line(net_cigre_lv, busC5, busC15, length_km=0.030, std_type='OH2',
                   name='Line C5-C15')
    pp.create_line(net_cigre_lv, busC15, busC16, length_km=0.030, std_type='OH2',
                   name='Line C15-C16')
    pp.create_line(net_cigre_lv, busC15, busC17, length_km=0.030, std_type='OH3',
                   name='Line C15-C17')
    pp.create_line(net_cigre_lv, busC16, busC18, length_km=0.030, std_type='OH3',
                   name='Line C16-C18')
    pp.create_line(net_cigre_lv, busC8, busC19, length_km=0.030, std_type='OH3',
                   name='Line C8-C19')
    pp.create_line(net_cigre_lv, busC9, busC20, length_km=0.030, std_type='OH3',
                   name='Line C9-C20')

    # Trafos
    pp.create_transformer_from_parameters(net_cigre_lv, busR0, busR1, sn_mva=0.5, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=1.0, vk_percent=4.123106,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo R0-R1')

    pp.create_transformer_from_parameters(net_cigre_lv, busI0, busI1, sn_mva=0.15, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=1.003125, vk_percent=4.126896,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo I0-I1')

    pp.create_transformer_from_parameters(net_cigre_lv, busC0, busC1, sn_mva=0.3, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=0.993750, vk_percent=4.115529,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo C0-C1')

    # External grid
    pp.create_ext_grid(net_cigre_lv, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0,
                       s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0)

    # Loads
    pp.create_load(net_cigre_lv, busR1, p_mw=0.0, q_mvar=0.062449980, name='Load R1')
    pp.create_load(net_cigre_lv, busR11, p_mw=0.01425, q_mvar=0.004683748, name='Load R11')
    pp.create_load(net_cigre_lv, busR15, p_mw=0.0494, q_mvar=0.016236995, name='Load R15')
    pp.create_load(net_cigre_lv, busR16, p_mw=0.05225, q_mvar=0.017173744, name='Load R16')
    pp.create_load(net_cigre_lv, busR17, p_mw=0.03325, q_mvar=0.010928746, name='Load R17')
    pp.create_load(net_cigre_lv, busR18, p_mw=0.04465, q_mvar=0.014675745, name='Load R18')
    pp.create_load(net_cigre_lv, busI2, p_mw=0.0850, q_mvar=0.052678269, name='Load I2')
    pp.create_load(net_cigre_lv, busC1, p_mw=0.0, q_mvar=0.052306787, name='Load C1')
    pp.create_load(net_cigre_lv, busC12, p_mw=0.018, q_mvar=0.008717798, name='Load C12')
    pp.create_load(net_cigre_lv, busC13, p_mw=0.018, q_mvar=0.008717798, name='Load C13')
    pp.create_load(net_cigre_lv, busC14, p_mw=0.0225, q_mvar=0.010897247, name='Load C14')
    pp.create_load(net_cigre_lv, busC17, p_mw=0.0225, q_mvar=0.010897247, name='Load C17')
    pp.create_load(net_cigre_lv, busC18, p_mw=0.0072, q_mvar=0.003487119, name='Load C18')
    pp.create_load(net_cigre_lv, busC19, p_mw=0.0144, q_mvar=0.006974238, name='Load C19')
    pp.create_load(net_cigre_lv, busC20, p_mw=0.0072, q_mvar=0.003487119, name='Load C20')

    # Switches
    pp.create_switch(net_cigre_lv, bus0, busR0, et='b', closed=True, type='CB', name='S1')
    pp.create_switch(net_cigre_lv, bus0, busI0, et='b', closed=True, type='CB', name='S2')
    pp.create_switch(net_cigre_lv, bus0, busC0, et='b', closed=True, type='CB', name='S3')

    # Distributed generators
    pp.create_sgen(net_cigre_lv, busR15, p_mw=0.0, q_mvar=0.0, name='MT 1', max_p_mw=0.03, min_p_mw=0, max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_sgen(net_cigre_lv, busR18, p_mw=0.0, q_mvar=0.0, name='FC 1', max_p_mw=0.05, min_p_mw=0, max_q_mvar=0.0375, min_q_mvar=-0.0375)
    pp.create_sgen(net_cigre_lv, busC17, p_mw=0.0, q_mvar=0.0, name='MT 2', max_p_mw=0.04, min_p_mw=0, max_q_mvar=0.03, min_q_mvar=-0.03)
    pp.create_sgen(net_cigre_lv, busR16, p_mw=0.01, type='PV', name='PV 1', max_p_mw=0.01, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net_cigre_lv, busR17, p_mw=0.01, type='PV', name='PV 2', max_p_mw=0.01, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net_cigre_lv, busC19, p_mw=0.01, type='PV', name='PV 3', max_p_mw=0.01, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net_cigre_lv, busR16, p_mw=0.01, type='WP', name='WP 1', max_p_mw=0.01, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net_cigre_lv, busC20, p_mw=0.01, type='WP', name='WP 2', max_p_mw=0.01, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)

    # storage
    pp.create_storage(net_cigre_lv, busR11, p_mw=0.1, max_e_mwh=0.5, sn_mva=0.1, soc_percent=50, min_e_mwh=0.05, name='Battery 1')
    pp.create_storage(net_cigre_lv, busC13, p_mw=0.06, max_e_mwh=0.3, sn_mva=0.1, soc_percent=50, min_e_mwh=0.03, name='Battery 2')

    # Bus geo data
    net_cigre_lv.bus_geodata = read_json(
        """{"x":{"0":0.2,"1":0.2,"2":-1.4583333333,"3":-1.4583333333,"4":-1.4583333333,
        "5":-1.9583333333,"6":-2.7083333333,"7":-2.7083333333,"8":-3.2083333333,"9":-3.2083333333,
        "10":-3.2083333333,"11":-3.7083333333,"12":-0.9583333333,"13":-1.2083333333,
        "14":-1.2083333333,"15":-1.2083333333,"16":-1.2083333333,"17":-2.2083333333,
        "18":-2.7083333333,"19":-3.7083333333,"20":0.2,"21":0.2,"22":0.2,"23":0.2,"24":1.9166666667,
        "25":1.9166666667,"26":1.9166666667,"27":0.5416666667,"28":0.5416666667,"29":-0.2083333333,
        "30":-0.2083333333,"31":-0.2083333333,"32":-0.7083333333,"33":3.2916666667,
        "34":2.7916666667,"35":2.2916666667,"36":3.2916666667,"37":3.7916666667,"38":1.2916666667,
        "39":0.7916666667,"40":1.7916666667,"41":0.7916666667,"42":0.2916666667,"43":-0.7083333333},
        "y":{"0":1.0,"1":1.0,"2":2.0,"3":3.0,"4":4.0,"5":5.0,"6":6.0,"7":7.0,"8":8.0,"9":9.0,
        "10":10.0,"11":11.0,"12":5.0,"13":6.0,"14":7.0,"15":8.0,"16":9.0,"17":8.0,"18":11.0,
        "19":12.0,"20":1.0,"21":2.0,"22":3.0,"23":1.0,"24":2.0,"25":3.0,"26":4.0,"27":5.0,"28":6.0,
        "29":7.0,"30":8.0,"31":9.0,"32":10.0,"33":5.0,"34":6.0,"35":7.0,"36":7.0,"37":6.0,"38":7.0,
        "39":8.0,"40":8.0,"41":9.0,"42":10.0,"43":11.0}}""")
    # Match bus.index
    net_cigre_lv.bus_geodata = net_cigre_lv.bus_geodata.loc[net_cigre_lv.bus.index]

    return net_cigre_lv


if __name__ == '__main__':
    net = CIGRE_LV()
    pp.runpp(net)
    print(net.res_bus)
    print(net.res_ext_grid)
    print(net.trafo)
    print(net.res_load['p_mw'].values.sum())
