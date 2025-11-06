import pandapower as pp

from powergrid.networks.lines import *


def IEEE13Bus(name=''):
    """
    Create the IEEE 13 bus from IEEE PES Test Feeders:
    "https://site.ieee.org/pes-testfeeders/resources/”.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net = pp.create_empty_network(name)
    if len(name) > 0: name += ' '

    # Linedata
    pp.create_std_type(net, CF601, name='CF-601', element='line')
    pp.create_std_type(net, CF602, name='CF-602', element='line')
    pp.create_std_type(net, CF603, name='CF-603', element='line')
    pp.create_std_type(net, CF604, name='CF-604', element='line')
    pp.create_std_type(net, CF605, name='CF-605', element='line')
    pp.create_std_type(net, CF606, name='CF-606', element='line')
    pp.create_std_type(net, CF607, name='CF-607', element='line')

    # Busses
    bus_650 = pp.create_bus(net, name='{}Bus 650'.format(name), vn_kv=24.9, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_646 = pp.create_bus(net, name='{}Bus 646'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_645 = pp.create_bus(net, name='{}Bus 645'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_632 = pp.create_bus(net, name='{}Bus 632'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_633 = pp.create_bus(net, name='{}Bus 633'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_634 = pp.create_bus(net, name='{}Bus 634'.format(name), vn_kv=0.48, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_611 = pp.create_bus(net, name='{}Bus 611'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_684 = pp.create_bus(net, name='{}Bus 684'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_671 = pp.create_bus(net, name='{}Bus 671'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_692 = pp.create_bus(net, name='{}Bus 692'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_675 = pp.create_bus(net, name='{}Bus 675'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_652 = pp.create_bus(net, name='{}Bus 652'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)
    bus_680 = pp.create_bus(net, name='{}Bus 680'.format(name), vn_kv=4.16, type='b', zone='13_BUS', max_vm_pu=1.05, min_vm_pu=0.95)

    # Lines
    pp.create_line(net, bus_632, bus_645, length_km=0.1524, std_type='CF-603', name='{}Line 1'.format(name))
    pp.create_line(net, bus_632, bus_633, length_km=0.1524, std_type='CF-602', name='{}Line 2'.format(name))
    pp.create_line(net, bus_645, bus_646, length_km=0.0914, std_type='CF-603', name='{}Line 3'.format(name))
    pp.create_line(net, bus_684, bus_652, length_km=0.2438, std_type='CF-607', name='{}Line 4'.format(name))
    pp.create_line(net, bus_632, bus_671, length_km=0.6096, std_type='CF-601', name='{}Line 5'.format(name))
    pp.create_line(net, bus_671, bus_684, length_km=0.0914, std_type='CF-604', name='{}Line 6'.format(name))
    pp.create_line(net, bus_671, bus_680, length_km=0.3048, std_type='CF-601', name='{}Line 7'.format(name))
    pp.create_line(net, bus_684, bus_611, length_km=0.0914, std_type='CF-605', name='{}Line 8'.format(name))
    pp.create_line(net, bus_692, bus_675, length_km=0.1524, std_type='CF-606', name='{}Line 9'.format(name))

    pp.create_switch(net, bus=bus_671,  element=bus_692, et='b', type="LS", z_ohm=0.0, name='{}Switch 1'.format(name))

    # Substation
    pp.create_transformer_from_parameters(net, bus_650, bus_632, sn_mva=5, vn_hv_kv=24.9,
                                          vn_lv_kv=4.16, vkr_percent=1.0, vk_percent=8.0,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          tap_side='lv', tap_neutral=0, tap_max=2, tap_min=-2, 
                                          tap_step_percent=2.5, tap_pos=0, tap_phase_shifter=False,
                                          name='{}Substation'.format(name))

    # Traformer
    pp.create_transformer_from_parameters(net, bus_633, bus_634, sn_mva=0.5, vn_hv_kv=4.16,
                                          vn_lv_kv=0.48, vkr_percent=1.1, vk_percent=2.0,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=0.0,
                                          name='{}Transformer 1'.format(name))

    # Loads
    pp.create_load(net, bus_634, p_mw=0.400, q_mvar=0.290, name='{}Load 634 spot'.format(name))
    pp.create_load(net, bus_645, p_mw=0.170, q_mvar=0.125, name='{}Load 645 spot'.format(name))
    pp.create_load(net, bus_646, p_mw=0.230, q_mvar=0.132, name='{}Load 646 spot'.format(name))
    pp.create_load(net, bus_652, p_mw=0.128, q_mvar=0.086, name='{}Load 652 spot'.format(name))
    pp.create_load(net, bus_671, p_mw=1.115, q_mvar=0.660, name='{}Load 671 spot'.format(name))
    pp.create_load(net, bus_675, p_mw=0.843, q_mvar=0.462, name='{}Load 675 spot'.format(name))
    pp.create_load(net, bus_692, p_mw=0.843, q_mvar=0.462, name='{}Load 692 spot'.format(name))
    pp.create_load(net, bus_611, p_mw=0.170, q_mvar=0.080, name='{}Load 611 spot'.format(name))

    # External grid
    pp.create_ext_grid(net, bus_650)

    return net

if __name__ == '__main__':
    net = IEEE13Bus()
    pp.runpp(net)
    print(net.res_bus)
    print(net.res_ext_grid)
    print(net.trafo)
    print(net.res_load['p_mw'].values.sum())
