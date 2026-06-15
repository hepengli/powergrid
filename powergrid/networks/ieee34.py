import pandapower as pp
from powergrid.networks.lines import *

def IEEE34Bus(name=''):
    """
    Create the IEEE 34 bus from IEEE PES Test Feeders:
    "https://site.ieee.org/pes-testfeeders/resources/”.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net = pp.create_empty_network(name)
    if len(name) > 0: name += ' '

    # Linedata
    pp.create_std_type(net, CF300, name='CF-300', element='line')
    pp.create_std_type(net, CF301, name='CF-301', element='line')
    pp.create_std_type(net, CF302, name='CF-302', element='line')
    pp.create_std_type(net, CF303, name='CF-303', element='line')
    pp.create_std_type(net, CF304, name='CF-304', element='line')

    # Busses
    # bus_0 = pp.create_bus(net, name='Bus 0', vn_kv=69.0, type='b', zone='34_BUS')
    bus_800 = pp.create_bus(net, name='{}Bus 800'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_802 = pp.create_bus(net, name='{}Bus 802'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_806 = pp.create_bus(net, name='{}Bus 806'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_808 = pp.create_bus(net, name='{}Bus 808'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_810 = pp.create_bus(net, name='{}Bus 810'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_812 = pp.create_bus(net, name='{}Bus 812'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_814 = pp.create_bus(net, name='{}Bus 814'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_RG10 = pp.create_bus(net, name='{}Bus RG10'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_850 = pp.create_bus(net, name='{}Bus 850'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_816 = pp.create_bus(net, name='{}Bus 816'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_818 = pp.create_bus(net, name='{}Bus 818'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_820 = pp.create_bus(net, name='{}Bus 820'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_822 = pp.create_bus(net, name='{}Bus 822'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_824 = pp.create_bus(net, name='{}Bus 824'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_826 = pp.create_bus(net, name='{}Bus 826'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_828 = pp.create_bus(net, name='{}Bus 828'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_830 = pp.create_bus(net, name='{}Bus 830'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_854 = pp.create_bus(net, name='{}Bus 854'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_852 = pp.create_bus(net, name='{}Bus 852'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_RG11 = pp.create_bus(net, name='{}Bus RG11'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_832 = pp.create_bus(net, name='{}Bus 832'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_858 = pp.create_bus(net, name='{}Bus 858'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_834 = pp.create_bus(net, name='{}Bus 834'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_842 = pp.create_bus(net, name='{}Bus 842'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_844 = pp.create_bus(net, name='{}Bus 844'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_846 = pp.create_bus(net, name='{}Bus 846'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_848 = pp.create_bus(net, name='{}Bus 848'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_860 = pp.create_bus(net, name='{}Bus 860'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_836 = pp.create_bus(net, name='{}Bus 836'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_840 = pp.create_bus(net, name='{}Bus 840'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_862 = pp.create_bus(net, name='{}Bus 862'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_838 = pp.create_bus(net, name='{}Bus 838'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_864 = pp.create_bus(net, name='{}Bus 864'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_888 = pp.create_bus(net, name='{}Bus 888'.format(name), vn_kv=4.16, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_890 = pp.create_bus(net, name='{}Bus 890'.format(name), vn_kv=4.16, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_856 = pp.create_bus(net, name='{}Bus 856'.format(name), vn_kv=24.9, type='b', zone='34_BUS', max_vm_pu=1.05, min_vm_pu=0.95)

    # Lines
    pp.create_line(net, bus_800, bus_802, length_km=0.786384, std_type='CF-300', name='{}Line 0'.format(name))
    pp.create_line(net, bus_802, bus_806, length_km=0.527304, std_type='CF-300', name='{}Line 1'.format(name))
    pp.create_line(net, bus_806, bus_808, length_km=9.823704, std_type='CF-300', name='{}Line 2'.format(name))
    pp.create_line(net, bus_808, bus_810, length_km=1.769059, std_type='CF-303', name='{}Line 3'.format(name))
    pp.create_line(net, bus_808, bus_812, length_km=11.43000, std_type='CF-300', name='{}Line 4'.format(name))
    pp.create_line(net, bus_812, bus_814, length_km=9.061704, std_type='CF-300', name='{}Line 5'.format(name))
    pp.create_line(net, bus_RG10, bus_850, length_km=0.003048, std_type='CF-301', name=' {}Line 6'.format(name))
    pp.create_line(net, bus_816, bus_818, length_km=0.521208, std_type='CF-302', name='{}Line 7'.format(name))
    pp.create_line(net, bus_816, bus_824, length_km=3.112008, std_type='CF-301', name='{}Line 8'.format(name))
    pp.create_line(net, bus_818, bus_820, length_km=14.67612, std_type='CF-302', name='{}Line 9'.format(name))
    pp.create_line(net, bus_820, bus_822, length_km=4.187952, std_type='CF-302', name='{}Line 10'.format(name))
    pp.create_line(net, bus_824, bus_826, length_km=0.923544, std_type='CF-303', name='{}Line 11'.format(name))
    pp.create_line(net, bus_824, bus_828, length_km=0.256032, std_type='CF-301', name='{}Line 12'.format(name))
    pp.create_line(net, bus_828, bus_830, length_km=6.230112, std_type='CF-301', name='{}Line 13'.format(name))
    pp.create_line(net, bus_830, bus_854, length_km=0.158496, std_type='CF-301', name='{}Line 14'.format(name))
    pp.create_line(net, bus_832, bus_858, length_km=1.493520, std_type='CF-301', name='{}Line 15'.format(name))
    pp.create_line(net, bus_834, bus_860, length_km=0.615696, std_type='CF-301', name='{}Line 16'.format(name))
    pp.create_line(net, bus_834, bus_842, length_km=0.085344, std_type='CF-301', name='{}Line 17'.format(name))
    pp.create_line(net, bus_836, bus_840, length_km=0.262128, std_type='CF-301', name='{}Line 18'.format(name))
    pp.create_line(net, bus_836, bus_862, length_km=0.085344, std_type='CF-301', name='{}Line 19'.format(name))
    pp.create_line(net, bus_842, bus_844, length_km=0.411480, std_type='CF-301', name='{}Line 20'.format(name))
    pp.create_line(net, bus_844, bus_846, length_km=1.109472, std_type='CF-301', name='{}Line 21'.format(name))
    pp.create_line(net, bus_846, bus_848, length_km=0.161544, std_type='CF-301', name='{}Line 22'.format(name))
    pp.create_line(net, bus_850, bus_816, length_km=0.094488, std_type='CF-301', name='{}Line 23'.format(name))
    pp.create_line(net, bus_RG11, bus_832, length_km=0.003048, std_type='CF-301', name='{}Line 24'.format(name))
    pp.create_line(net, bus_854, bus_856, length_km=7.110984, std_type='CF-303', name='{}Line 25'.format(name))
    pp.create_line(net, bus_854, bus_852, length_km=11.22578, std_type='CF-301', name='{}Line 26'.format(name))
    pp.create_line(net, bus_858, bus_864, length_km=0.493776, std_type='CF-302', name='{}Line 27'.format(name))
    pp.create_line(net, bus_858, bus_834, length_km=1.776984, std_type='CF-301', name='{}Line 28'.format(name))
    pp.create_line(net, bus_860, bus_836, length_km=0.816864, std_type='CF-301', name='{}Line 29'.format(name))
    pp.create_line(net, bus_860, bus_838, length_km=1.481328, std_type='CF-304', name='{}Line 30'.format(name))
    pp.create_line(net, bus_888, bus_890, length_km=3.218688, std_type='CF-300', name='{}Line 31'.format(name))

    # Regulator 1
    pp.create_transformer_from_parameters(net, bus_814, bus_RG10, sn_mva=2.5, vn_hv_kv=24.9,
                                          vn_lv_kv=24.9, vkr_percent=0.320088*2.5, vk_percent=0.357539*2.5,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          tap_side='lv', tap_neutral=0, tap_max=16, tap_min=-16, 
                                          tap_step_percent=0.625, tap_pos=12, tap_phase_shifter=False,
                                          name='{}Regulator1'.format(name))
    # Regulator 2
    pp.create_transformer_from_parameters(net, bus_852, bus_RG11, sn_mva=2.5, vn_hv_kv=24.9,
                                          vn_lv_kv=24.9, vkr_percent=0.320088*2.5, vk_percent=0.357539*2.5,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          tap_side='lv', tap_neutral=0, tap_max=16, tap_min=-16, 
                                          tap_step_percent=0.625, tap_pos=16, tap_phase_shifter=False,
                                          name='{}Regulator2'.format(name))
    # # Substation
    # pp.create_transformer_from_parameters(net, bus_0, bus_800, sn_mva=2.5, vn_hv_kv=69.0,
    #                                       vn_lv_kv=24.9, vkr_percent=1.0, vk_percent=8.062257,
    #                                       pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
    #                                     #   tap_side='lv', tap_neutral=0, tap_max=2, tap_min=-2, 
    #                                     #   tap_step_percent=2.5, tap_pos=0, tap_phase_shifter=False,
    #                                       name='{}Substation'.format(name))
    # Traformer
    pp.create_transformer_from_parameters(net, bus_832, bus_888, sn_mva=0.5, vn_hv_kv=24.9,
                                          vn_lv_kv=4.16, vkr_percent=1.9, vk_percent=4.5,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          name='{}Transformer1'.format(name))

    # Loads
    pp.create_load(net, bus_806, p_mw=0.055, q_mvar=0.029, name='{}Load 806'.format(name))
    pp.create_load(net, bus_810, p_mw=0.016, q_mvar=0.008, name='{}Load 810'.format(name))
    pp.create_load(net, bus_820, p_mw=0.034, q_mvar=0.017, name='{}Load 820'.format(name))
    pp.create_load(net, bus_822, p_mw=0.135, q_mvar=0.070, name='{}Load 822'.format(name))
    pp.create_load(net, bus_824, p_mw=0.005, q_mvar=0.002, name='{}Load 824'.format(name))
    pp.create_load(net, bus_826, p_mw=0.004, q_mvar=0.020, name='{}Load 826'.format(name))
    pp.create_load(net, bus_828, p_mw=0.004, q_mvar=0.002, name='{}Load 828'.format(name))
    pp.create_load(net, bus_830, p_mw=0.007, q_mvar=0.003, name='{}Load 830'.format(name))
    pp.create_load(net, bus_856, p_mw=0.004, q_mvar=0.002, name='{}Load 856'.format(name))
    pp.create_load(net, bus_858, p_mw=0.015, q_mvar=0.007, name='{}Load 858'.format(name))
    pp.create_load(net, bus_864, p_mw=0.002, q_mvar=0.001, name='{}Load 864'.format(name))
    pp.create_load(net, bus_834, p_mw=0.032, q_mvar=0.017, name='{}Load 834'.format(name))
    pp.create_load(net, bus_860, p_mw=0.029, q_mvar=0.073, name='{}Load 860'.format(name))
    pp.create_load(net, bus_836, p_mw=0.082, q_mvar=0.043, name='{}Load 836'.format(name))
    pp.create_load(net, bus_840, p_mw=0.040, q_mvar=0.020, name='{}Load 840'.format(name))
    pp.create_load(net, bus_838, p_mw=0.028, q_mvar=0.014, name='{}Load 838'.format(name))
    pp.create_load(net, bus_844, p_mw=0.009, q_mvar=0.005, name='{}Load 844'.format(name))
    pp.create_load(net, bus_846, p_mw=0.037, q_mvar=0.031, name='{}Load 846'.format(name))
    pp.create_load(net, bus_848, p_mw=0.023, q_mvar=0.011, name='{}Load 848'.format(name))
 
    pp.create_load(net, bus_860, p_mw=0.060, q_mvar=0.048, name='{}Load 860 spot'.format(name))
    pp.create_load(net, bus_840, p_mw=0.027, q_mvar=0.021, name='{}Load 840 spot'.format(name))
    pp.create_load(net, bus_844, p_mw=0.405, q_mvar=0.315, name='{}Load 844 spot'.format(name))
    pp.create_load(net, bus_848, p_mw=0.060, q_mvar=0.048, name='{}Load 848 spot'.format(name))
    pp.create_load(net, bus_890, p_mw=0.450, q_mvar=0.225, name='{}Load 890 spot'.format(name))
    pp.create_load(net, bus_830, p_mw=0.045, q_mvar=0.020, name='{}Load 830 spot'.format(name))

    # External grid
    pp.create_ext_grid(net, bus_800, vm_pu=1.05, va_degree=0.0)

    # # Distributed generators
    # pp.create_sgen(net, bus_848, p_mw=0., q_mvar=0., name='DG 1', max_p_mw=0.66, min_p_mw=0, max_q_mvar=0.5, min_q_mvar=0)
    # pp.create_sgen(net, bus_890, p_mw=0., q_mvar=0., name='DG 2', max_p_mw=0.50, min_p_mw=0, max_q_mvar=0.375, min_q_mvar=0)
    # pp.create_sgen(net, bus_822, p_mw=0., type='PV', name='PV 1', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    # pp.create_sgen(net, bus_856, p_mw=0., type='PV', name='PV 2', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    # pp.create_sgen(net, bus_838, p_mw=0., type='PV', name='PV 3', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    # pp.create_sgen(net, bus_822, p_mw=0., type='WP', name='WP 1', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    # pp.create_sgen(net, bus_826, p_mw=0., type='WP', name='WP 2', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)
    # pp.create_sgen(net, bus_838, p_mw=0., type='WP', name='WP 3', max_p_mw=0.1, min_p_mw=0, max_q_mvar=0, min_q_mvar=0)

    # Shunt capacity bank
    pp.create_shunt_as_capacitor(net, bus_844, q_mvar=-0.30, name='SCB1', step=0, max_step=4, loss_factor=0.0)
    pp.create_shunt_as_capacitor(net, bus_848, q_mvar=-0.45, name='SCB2', step=0, max_step=4, loss_factor=0.0)

    # # # storage
    # pp.create_storage(net, bus_810, p_mw=0.0, max_e_mwh=2, sn_mva=1.0, soc_percent=50, min_e_mwh=0.2, name='Storage')

    return net

if __name__ == '__main__':
    from powergrid.networks.ieee34 import IEEE34Bus
    import pandapower as pp
    net = IEEE34Bus()
    pp.runpp(net)
    print(net.res_bus)
    print(net.res_ext_grid)
    print(net.trafo)
    print(net.res_load['p_mw'].values.sum())