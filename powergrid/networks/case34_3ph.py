from pandas import read_json
from numpy import nan
import pandapower as pp
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

def case34_3ph():
    """
    Create the IEEE 34 bus from IEEE PES Test Feeders:
    "https://site.ieee.org/pes-testfeeders/resources/”.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net = pp.create_empty_network()

    # Linedata
    # CF-300
    line_data = {'c_nf_per_km': 3.8250977, 'r_ohm_per_km': 0.69599766,
                 'x_ohm_per_km': 0.5177677, 
                 'c0_nf_per_km': 1.86976748, 'r0_ohm_per_km': 1.08727498,
                 'x0_ohm_per_km': 1.47374703, 
                 'max_i_ka': 0.23, 'type': 'ol'}
    pp.create_std_type(net, line_data, name='CF-300', element='line')

    # CF-301
    line_data = {'c_nf_per_km': 3.66884364, 'r_ohm_per_km': 1.05015841,
                 'x_ohm_per_km': 0.52265586, 
                 'c0_nf_per_km': 1.82231544, 'r0_ohm_per_km': 1.48350255,
                 'x0_ohm_per_km': 1.60203942, 
                 'max_i_ka': 0.18, 'type': 'ol'}
    pp.create_std_type(net, line_data, name='CF-301', element='line')

    # CF-302
    line_data = {'c_nf_per_km': 0.8751182, 'r_ohm_per_km': 0.5798427,
                 'x_ohm_per_km': 0.30768221, 
                 'c0_nf_per_km': 0.8751182, 'r0_ohm_per_km': 0.5798427,
                 'x0_ohm_per_km': 0.30768221, 
                 'max_i_ka': 0.14, 'type': 'ol'}
    pp.create_std_type(net, line_data, name='CF-302', element='line')

    # CF-303
    line_data = {'c_nf_per_km': 0.8751182, 'r_ohm_per_km': 0.5798427,
                 'x_ohm_per_km': 0.30768221, 
                 'c0_nf_per_km': 0.8751182, 'r0_ohm_per_km': 0.5798427,
                 'x0_ohm_per_km': 0.30768221, 
                 'max_i_ka': 0.14, 'type': 'ol'}
    pp.create_std_type(net, line_data, name='CF-303', element='line')

    # CF-304
    line_data = {'c_nf_per_km': 0.90382554, 'r_ohm_per_km': 0.39802955,
                 'x_ohm_per_km': 0.29436416, 
                 'c0_nf_per_km': 0.90382554, 'r0_ohm_per_km': 0.39802955,
                 'x0_ohm_per_km': 0.29436416, 
                 'max_i_ka': 0.18, 'type': 'ol'}
    pp.create_std_type(net, line_data, name='CF-304', element='line')

    # Busses
    bus0 = pp.create_bus(net, name='Bus 0', vn_kv=24.9, type='n', zone='34_BUS')
    bus_800 = pp.create_bus(net, name='Bus 800', vn_kv=24.9, type='n', zone='34_BUS')
    bus_802 = pp.create_bus(net, name='Bus 802', vn_kv=24.9, type='n', zone='34_BUS')
    bus_806 = pp.create_bus(net, name='Bus 806', vn_kv=24.9, type='n', zone='34_BUS')
    bus_808 = pp.create_bus(net, name='Bus 808', vn_kv=24.9, type='n', zone='34_BUS')
    bus_810 = pp.create_bus(net, name='Bus 810', vn_kv=24.9, type='n', zone='34_BUS')
    bus_812 = pp.create_bus(net, name='Bus 812', vn_kv=24.9, type='n', zone='34_BUS')
    bus_814 = pp.create_bus(net, name='Bus 814', vn_kv=24.9, type='n', zone='34_BUS')
    bus_850 = pp.create_bus(net, name='Bus 850', vn_kv=24.9, type='n', zone='34_BUS')
    bus_816 = pp.create_bus(net, name='Bus 816', vn_kv=24.9, type='n', zone='34_BUS')
    bus_818 = pp.create_bus(net, name='Bus 818', vn_kv=24.9, type='n', zone='34_BUS')
    bus_820 = pp.create_bus(net, name='Bus 820', vn_kv=24.9, type='n', zone='34_BUS')
    bus_822 = pp.create_bus(net, name='Bus 822', vn_kv=24.9, type='n', zone='34_BUS')
    bus_824 = pp.create_bus(net, name='Bus 824', vn_kv=24.9, type='n', zone='34_BUS')
    bus_826 = pp.create_bus(net, name='Bus 826', vn_kv=24.9, type='n', zone='34_BUS')
    bus_828 = pp.create_bus(net, name='Bus 828', vn_kv=24.9, type='n', zone='34_BUS')
    bus_830 = pp.create_bus(net, name='Bus 830', vn_kv=24.9, type='n', zone='34_BUS')
    bus_854 = pp.create_bus(net, name='Bus 854', vn_kv=24.9, type='n', zone='34_BUS')
    bus_852 = pp.create_bus(net, name='Bus 852', vn_kv=24.9, type='n', zone='34_BUS')
    bus_832 = pp.create_bus(net, name='Bus 832', vn_kv=24.9, type='n', zone='34_BUS')
    bus_858 = pp.create_bus(net, name='Bus 858', vn_kv=24.9, type='n', zone='34_BUS')
    bus_834 = pp.create_bus(net, name='Bus 834', vn_kv=24.9, type='n', zone='34_BUS')
    bus_842 = pp.create_bus(net, name='Bus 842', vn_kv=24.9, type='n', zone='34_BUS')
    bus_844 = pp.create_bus(net, name='Bus 844', vn_kv=24.9, type='n', zone='34_BUS')
    bus_846 = pp.create_bus(net, name='Bus 846', vn_kv=24.9, type='n', zone='34_BUS')
    bus_848 = pp.create_bus(net, name='Bus 848', vn_kv=24.9, type='n', zone='34_BUS')
    bus_860 = pp.create_bus(net, name='Bus 860', vn_kv=24.9, type='n', zone='34_BUS')
    bus_836 = pp.create_bus(net, name='Bus 836', vn_kv=24.9, type='n', zone='34_BUS')
    bus_840 = pp.create_bus(net, name='Bus 840', vn_kv=24.9, type='n', zone='34_BUS')
    bus_862 = pp.create_bus(net, name='Bus 862', vn_kv=24.9, type='n', zone='34_BUS')
    bus_838 = pp.create_bus(net, name='Bus 838', vn_kv=24.9, type='n', zone='34_BUS')
    bus_864 = pp.create_bus(net, name='Bus 864', vn_kv=24.9, type='n', zone='34_BUS')
    bus_888 = pp.create_bus(net, name='Bus 888', vn_kv=4.16, type='n', zone='34_BUS')
    bus_890 = pp.create_bus(net, name='Bus 890', vn_kv=4.16, type='n', zone='34_BUS')
    bus_856 = pp.create_bus(net, name='Bus 856', vn_kv=24.9, type='n', zone='34_BUS')

    # Lines
    pp.create_line(net, bus_800, bus_802, length_km=0.786384, std_type='CF-300',
                   name='Line 0')
    pp.create_line(net, bus_802, bus_806, length_km=0.527304, std_type='CF-300',
                   name='Line 1')
    pp.create_line(net, bus_806, bus_808, length_km=9.823704, std_type='CF-300',
                   name='Line 2')
    pp.create_line(net, bus_808, bus_810, length_km=1.769059, std_type='CF-303',
                   name='Line 3')
    pp.create_line(net, bus_808, bus_812, length_km=11.43000, std_type='CF-300',
                   name='Line 4')
    pp.create_line(net, bus_812, bus_814, length_km=9.061704, std_type='CF-300',
                   name='Line 5')
    # pp.create_line(net, bus_814, bus_850, length_km=0.003048, std_type='CF-301',
    #                name='Line 6')
    pp.create_line(net, bus_816, bus_818, length_km=0.521208, std_type='CF-302',
                   name='Line 7')
    pp.create_line(net, bus_816, bus_824, length_km=3.112008, std_type='CF-301',
                   name='Line 8')
    pp.create_line(net, bus_818, bus_820, length_km=14.67612, std_type='CF-302',
                   name='Line 9')
    pp.create_line(net, bus_820, bus_822, length_km=4.187952, std_type='CF-302',
                   name='Line 10')
    pp.create_line(net, bus_824, bus_826, length_km=0.923544, std_type='CF-303',
                   name='Line 11')
    pp.create_line(net, bus_824, bus_828, length_km=0.256032, std_type='CF-301',
                   name='Line 12')
    pp.create_line(net, bus_828, bus_830, length_km=6.230112, std_type='CF-301',
                   name='Line 13')
    pp.create_line(net, bus_830, bus_854, length_km=0.158496, std_type='CF-301',
                   name='Line 14')
    pp.create_line(net, bus_832, bus_858, length_km=1.493520, std_type='CF-301',
                   name='Line 15')
    pp.create_line(net, bus_834, bus_860, length_km=0.615696, std_type='CF-301',
                   name='Line 16')
    pp.create_line(net, bus_834, bus_842, length_km=0.085344, std_type='CF-301',
                   name='Line 17')
    pp.create_line(net, bus_836, bus_840, length_km=0.262128, std_type='CF-301',
                   name='Line 18')
    pp.create_line(net, bus_836, bus_862, length_km=0.085344, std_type='CF-301',
                   name='Line 19')
    pp.create_line(net, bus_842, bus_844, length_km=0.411480, std_type='CF-301',
                   name='Line 20')
    pp.create_line(net, bus_844, bus_846, length_km=1.109472, std_type='CF-301',
                   name='Line 21')
    pp.create_line(net, bus_846, bus_848, length_km=0.161544, std_type='CF-301',
                   name='Line 22')
    pp.create_line(net, bus_850, bus_816, length_km=0.094488, std_type='CF-301',
                   name='Line 23')
    # pp.create_line(net, bus_852, bus_832, length_km=0.003048, std_type='CF-301',
    #                name='Line 24')
    pp.create_line(net, bus_854, bus_856, length_km=7.110984, std_type='CF-303',
                   name='Line 25')
    pp.create_line(net, bus_854, bus_852, length_km=11.22578, std_type='CF-301',
                   name='Line 26')
    pp.create_line(net, bus_858, bus_864, length_km=0.493776, std_type='CF-302',
                   name='Line 27')
    pp.create_line(net, bus_858, bus_834, length_km=1.776984, std_type='CF-301',
                   name='Line 28')
    pp.create_line(net, bus_860, bus_836, length_km=0.816864, std_type='CF-301',
                   name='Line 29')
    pp.create_line(net, bus_860, bus_838, length_km=1.481328, std_type='CF-304',
                   name='Line 30')
    pp.create_line(net, bus_888, bus_890, length_km=3.218688, std_type='CF-300',
                   name='Line 31')

    # Substation
    pp.create_transformer_from_parameters(net, bus0, bus_800, sn_mva=2.5, vn_hv_kv=24.9,
                                          vn_lv_kv=24.9, vkr_percent=1.0, vk_percent=8.062257,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          vkr0_percent=1.0, vk0_percent=8.062257, vector_group='YNyn',
                                          mag0_percent=100.0, mag0_rx=0, si0_hv_partial=0.9, 
                                          tap_side='lv', tap_neutral=0, tap_max=2, tap_min=-2, 
                                          tap_step_percent=2.5, tap_pos=-2,
                                          name='Substation')
    # Regulator 1
    pp.create_transformer_from_parameters(net, bus_814, bus_850, sn_mva=1, vn_hv_kv=24.9,
                                          vn_lv_kv=24.9, vkr_percent=0.320088, vk_percent=0.357539,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          vkr0_percent=0.452171, vk0_percent=0.665505, vector_group='YNyn',
                                          mag0_percent=100.0, mag0_rx=0, si0_hv_partial=0.9, 
                                          tap_side='lv', tap_neutral=0, tap_max=16, tap_min=-16, 
                                          tap_step_percent=0.625, tap_pos=0, tap_phase_shifter=False,
                                          name='Regulator 1')

    # Regulator 2
    pp.create_transformer_from_parameters(net, bus_852, bus_832, sn_mva=1, vn_hv_kv=24.9,
                                          vn_lv_kv=24.9, vkr_percent=0.320088, vk_percent=0.357539,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          vkr0_percent=0.452171, vk0_percent=0.665505, vector_group='YNyn',
                                          mag0_percent=100.0, mag0_rx=0, si0_hv_partial=0.9, 
                                          tap_side='lv', tap_neutral=0, tap_max=16, tap_min=-16, 
                                          tap_step_percent=0.625, tap_pos=0, tap_phase_shifter=False,
                                          name='Regulator 2')
    # Traformer
    pp.create_transformer_from_parameters(net, bus_832, bus_888, sn_mva=0.5, vn_hv_kv=24.9,
                                          vn_lv_kv=4.16, vkr_percent=1.9, vk_percent=4.5,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          vkr0_percent=1.9, vk0_percent=4.5, vector_group='YNyn',
                                          mag0_percent=100.0, mag0_rx=0, si0_hv_partial=0.9, 
                                          name='Traformer')


    # Loads
    pp.create_asymmetric_load(net, bus_806, p_a_mw=0, p_b_mw=0.03, p_c_mw=0.025, 
                              q_a_mvar=0, q_b_mvar=0.015, q_c_mvar=0.014, name='Load 806', type='wye')
    pp.create_asymmetric_load(net, bus_810, p_a_mw=0, p_b_mw=0.016, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.008, q_c_mvar=0, name='Load 810', type='wye')
    pp.create_asymmetric_load(net, bus_820, p_a_mw=0.034, p_b_mw=0, p_c_mw=0, 
                              q_a_mvar=0.017, q_b_mvar=0, q_c_mvar=0, name='Load 820', type='wye')
    pp.create_asymmetric_load(net, bus_822, p_a_mw=0.135, p_b_mw=0, p_c_mw=0, 
                              q_a_mvar=0.07, q_b_mvar=0, q_c_mvar=0, name='Load 822', type='wye')
    pp.create_asymmetric_load(net, bus_824, p_a_mw=0, p_b_mw=0.005, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.002, q_c_mvar=0, name='Load 824', type='delta')
    pp.create_asymmetric_load(net, bus_826, p_a_mw=0, p_b_mw=0.04, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.02, q_c_mvar=0, name='Load 826', type='wye')
    pp.create_asymmetric_load(net, bus_828, p_a_mw=0, p_b_mw=0, p_c_mw=0.004, 
                              q_a_mvar=0, q_b_mvar=0, q_c_mvar=0.002, name='Load 828', type='wye')
    pp.create_asymmetric_load(net, bus_830, p_a_mw=0.007, p_b_mw=0, p_c_mw=0, 
                              q_a_mvar=0.003, q_b_mvar=0, q_c_mvar=0, name='Load 830', type='wye')
    pp.create_asymmetric_load(net, bus_856, p_a_mw=0, p_b_mw=0.004, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.002, q_c_mvar=0, name='Load 856', type='wye')
    pp.create_asymmetric_load(net, bus_858, p_a_mw=0.007, p_b_mw=0.002, p_c_mw=0.006, 
                              q_a_mvar=0.003, q_b_mvar=0.001, q_c_mvar=0.003, name='Load 858', type='delta')
    pp.create_asymmetric_load(net, bus_864, p_a_mw=0.002, p_b_mw=0, p_c_mw=0, 
                              q_a_mvar=0.001, q_b_mvar=0, q_c_mvar=0, name='Load 864', type='wye')
    pp.create_asymmetric_load(net, bus_834, p_a_mw=0.004, p_b_mw=0.015, p_c_mw=0.013, 
                              q_a_mvar=0.002, q_b_mvar=0.008, q_c_mvar=0.007, name='Load 834', type='delta')
    pp.create_asymmetric_load(net, bus_860, p_a_mw=0.016, p_b_mw=0.02, p_c_mw=0.11, 
                              q_a_mvar=0.008, q_b_mvar=0.01, q_c_mvar=0.055, name='Load 860', type='delta')
    pp.create_asymmetric_load(net, bus_836, p_a_mw=0.03, p_b_mw=0.01, p_c_mw=0.042, 
                              q_a_mvar=0.015, q_b_mvar=0.006, q_c_mvar=0.022, name='Load 836', type='delta')
    pp.create_asymmetric_load(net, bus_840, p_a_mw=0.018, p_b_mw=0.022, p_c_mw=0, 
                              q_a_mvar=0.009, q_b_mvar=0.011, q_c_mvar=0, name='Load 840', type='delta')
    pp.create_asymmetric_load(net, bus_838, p_a_mw=0, p_b_mw=0.028, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.014, q_c_mvar=0, name='Load 838', type='wye')
    pp.create_asymmetric_load(net, bus_844, p_a_mw=0.009, p_b_mw=0, p_c_mw=0, 
                              q_a_mvar=0.005, q_b_mvar=0, q_c_mvar=0, name='Load 844', type='wye')
    pp.create_asymmetric_load(net, bus_846, p_a_mw=0, p_b_mw=0.025, p_c_mw=0.012, 
                              q_a_mvar=0, q_b_mvar=0.02, q_c_mvar=0.011, name='Load 846', type='wye')
    pp.create_asymmetric_load(net, bus_848, p_a_mw=0, p_b_mw=0.023, p_c_mw=0, 
                              q_a_mvar=0, q_b_mvar=0.011, q_c_mvar=0, name='Load 848', type='wye')

    pp.create_asymmetric_load(net, bus_860, p_a_mw=0.02, p_b_mw=0.02, p_c_mw=0.02, 
                              q_a_mvar=0.016, q_b_mvar=0.016, q_c_mvar=0.016, name='Load 860 spot', type='wye')
    pp.create_asymmetric_load(net, bus_840, p_a_mw=0.009, p_b_mw=0.009, p_c_mw=0.009, 
                              q_a_mvar=0.007, q_b_mvar=0.007, q_c_mvar=0.007, name='Load 840 spot', type='wye')
    pp.create_asymmetric_load(net, bus_844, p_a_mw=0.135, p_b_mw=0.135, p_c_mw=0.135, 
                              q_a_mvar=0.105, q_b_mvar=0.105, q_c_mvar=0.105, name='Load 844 spot', type='wye')
    pp.create_asymmetric_load(net, bus_848, p_a_mw=0.02, p_b_mw=0.02, p_c_mw=0.02, 
                              q_a_mvar=0.016, q_b_mvar=0.016, q_c_mvar=0.016, name='Load 848 spot', type='delta')
    pp.create_asymmetric_load(net, bus_890, p_a_mw=0.15, p_b_mw=0.15, p_c_mw=0.15, 
                              q_a_mvar=0.075, q_b_mvar=0.075, q_c_mvar=0.075, name='Load 890 spot', type='delta')
    pp.create_asymmetric_load(net, bus_830, p_a_mw=0.01, p_b_mw=0.01, p_c_mw=0.025, 
                              q_a_mvar=0.005, q_b_mvar=0.005, q_c_mvar=0.01, name='Load 830 spot', type='delta')

    # External grid
    pp.create_ext_grid(net, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=10.0,
                       s_sc_min_mva=10.0, rx_max=1, rx_min=1, r0x0_max=1, x0x_max=1)

    # Distributed generators
    pp.create_sgen(net, bus_848, p_mw=0.66, q_mvar=0.500, name='DG 1', max_p_mw=0.66, min_p_mw=0, max_q_mvar=0.5, min_q_mvar=0)
    pp.create_sgen(net, bus_890, p_mw=0.50, q_mvar=0.375, name='DG 2', max_p_mw=0.50, min_p_mw=0, max_q_mvar=0.375, min_q_mvar=0)
    pp.create_sgen(net, bus_822, p_mw=0.2, type='PV', name='PV 1', max_p_mw=0.2, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net, bus_856, p_mw=0.2, type='PV', name='PV 2', max_p_mw=0.2, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net, bus_838, p_mw=0.2, type='PV', name='PV 3', max_p_mw=0.2, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net, bus_822, p_mw=0.3, type='WP', name='WP 1', max_p_mw=0.3, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net, bus_826, p_mw=0.3, type='WP', name='WP 2', max_p_mw=0.3, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    pp.create_sgen(net, bus_838, p_mw=0.3, type='WP', name='WP 3', max_p_mw=0.3, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)

    # Shunt capacity bank
    pp.create_shunt(net, bus_840, q_mvar=-0.12, name='SCB 1', step=4, max_step=4)
    pp.create_shunt(net, bus_864, q_mvar=-0.12, name='SCB 2', step=4, max_step=4)

    # storage
    pp.create_storage(net, bus_810, p_mw=0.2, max_e_mwh=1.0, sn_mva=1.0, soc_percent=50, min_e_mwh=0.2, name='Storage')

    pp.add_zero_impedance_parameters(net)

    return net

# net = case34_3ph()
# pp.pf.runpp_3ph.runpp_3ph(net)
# print(net.res_bus_3ph)
# print(net.sn_mva)