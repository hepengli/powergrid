
import io
from pandas import read_json
from numpy import nan
import pandapower as pp
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

def create_cigre_network_mv(with_der="all"):
    """
    Create the CIGRE MV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OPTIONAL:
        **with_der** (boolean or str, False) - Range of DER consideration, which should be in
            (False, "pv_wind", "all"). The DER types, dimensions and locations are taken from CIGRE
            CaseStudy: "DER in Medium Voltage Systems"

    OUTPUT:
        **net** - The pandapower format network.
    """
    if with_der is True:
        raise ValueError("'with_der=True' is deprecated. Please use 'with_der=pv_wind'")
    if with_der not in [False, "pv_wind", "all"]:
        raise ValueError("'with_der' is unknown. It should be in [False, 'pv_wind', 'all'].")

    net_cigre_mv = pp.create_empty_network()

    # Linedata
    line_data = {'c_nf_per_km': 151.1749, 'r_ohm_per_km': 0.501,
                 'x_ohm_per_km': 0.716, 'max_i_ka': 0.145,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_mv, line_data, name='CABLE_CIGRE_MV', element='line')

    line_data = {'c_nf_per_km': 10.09679, 'r_ohm_per_km': 0.510,
                 'x_ohm_per_km': 0.366, 'max_i_ka': 0.195,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_mv, line_data, name='OHL_CIGRE_MV', element='line')

    # Busses
    # bus0 = pp.create_bus(net_cigre_mv, name='Bus 0', vn_kv=110, type='b', zone='CIGRE_MV')
    buses = pp.create_buses(net_cigre_mv, 14, name=['Bus %i' % i for i in range(1, 15)], vn_kv=20,
                            type='b', zone='CIGRE_MV')

    # Lines
    pp.create_line(net_cigre_mv, buses[0], buses[1], length_km=2.82,
                   std_type='CABLE_CIGRE_MV', name='Line 1-2')
    pp.create_line(net_cigre_mv, buses[1], buses[2], length_km=4.42,
                   std_type='CABLE_CIGRE_MV', name='Line 2-3')
    pp.create_line(net_cigre_mv, buses[2], buses[3], length_km=0.61,
                   std_type='CABLE_CIGRE_MV', name='Line 3-4')
    pp.create_line(net_cigre_mv, buses[3], buses[4], length_km=0.56,
                   std_type='CABLE_CIGRE_MV', name='Line 4-5')
    pp.create_line(net_cigre_mv, buses[4], buses[5], length_km=1.54,
                   std_type='CABLE_CIGRE_MV', name='Line 5-6')
    pp.create_line(net_cigre_mv, buses[7], buses[6], length_km=1.67,
                   std_type='CABLE_CIGRE_MV', name='Line 8-7')
    pp.create_line(net_cigre_mv, buses[7], buses[8], length_km=0.32,
                   std_type='CABLE_CIGRE_MV', name='Line 8-9')
    pp.create_line(net_cigre_mv, buses[8], buses[9], length_km=0.77,
                   std_type='CABLE_CIGRE_MV', name='Line 9-10')
    pp.create_line(net_cigre_mv, buses[9], buses[10], length_km=0.33,
                   std_type='CABLE_CIGRE_MV', name='Line 10-11')
    pp.create_line(net_cigre_mv, buses[2], buses[7], length_km=1.3,
                   std_type='CABLE_CIGRE_MV', name='Line 3-8')
    pp.create_line(net_cigre_mv, buses[11], buses[12], length_km=4.89,
                   std_type='OHL_CIGRE_MV', name='Line 12-13')
    pp.create_line(net_cigre_mv, buses[12], buses[13], length_km=2.99,
                   std_type='OHL_CIGRE_MV', name='Line 13-14')

    # line6_7 = pp.create_line(net_cigre_mv, buses[5], buses[6], length_km=0.24,
    #                          std_type='CABLE_CIGRE_MV', name='Line 6-7')
    # line4_11 = pp.create_line(net_cigre_mv, buses[10], buses[3], length_km=0.49,
    #                           std_type='CABLE_CIGRE_MV', name='Line 11-4')
    # line8_14 = pp.create_line(net_cigre_mv, buses[13], buses[7], length_km=2.,
    #                           std_type='OHL_CIGRE_MV', name='Line 14-8')

    # Ext-Grid
    pp.create_ext_grid(net_cigre_mv, buses[0], vm_pu=1.03, va_degree=0.,
                       s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1)

    pp.create_line(net_cigre_mv, buses[0], buses[11], length_km=0.001,
                   std_type='CABLE_CIGRE_MV', name='Line 1-11')

    # # Trafos
    # trafo0 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[0], sn_mva=25,
    #                                                vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
    #                                                vk_percent=12.00107, pfe_kw=0, i0_percent=0,
    #                                                shift_degree=30.0, name='Trafo 0-1')
    # trafo1 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[11], sn_mva=25,
    #                                                vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
    #                                                vk_percent=12.00107, pfe_kw=0, i0_percent=0,
    #                                                shift_degree=30.0, name='Trafo 0-12')

    # # Switches
    # # S2
    # pp.create_switch(net_cigre_mv, buses[5], line6_7, et='l', closed=True, type='LBS')
    # pp.create_switch(net_cigre_mv, buses[6], line6_7, et='l', closed=False, type='LBS', name='S2')
    # # S3
    # pp.create_switch(net_cigre_mv, buses[3], line4_11, et='l', closed=False, type='LBS', name='S3')
    # pp.create_switch(net_cigre_mv, buses[10], line4_11, et='l', closed=True, type='LBS')
    # # S1
    # pp.create_switch(net_cigre_mv, buses[7], line8_14, et='l', closed=False, type='LBS', name='S1')
    # pp.create_switch(net_cigre_mv, buses[13], line8_14, et='l', closed=True, type='LBS')
    # # trafos
    # pp.create_switch(net_cigre_mv, bus0, trafo0, et='t', closed=True, type='CB')
    # pp.create_switch(net_cigre_mv, bus0, trafo1, et='t', closed=True, type='CB')

    # Loads
    # Residential
    # pp.create_load_from_cosphi(net_cigre_mv, buses[0], 15.3, 0.98, "underexcited", name='Load R1')
    pp.create_load_from_cosphi(net_cigre_mv, buses[2], 0.285, 0.97, "underexcited", name='Load R3')
    pp.create_load_from_cosphi(net_cigre_mv, buses[3], 0.445, 0.97, "underexcited", name='Load R4')
    pp.create_load_from_cosphi(net_cigre_mv, buses[4], 0.750, 0.97, "underexcited", name='Load R5')
    pp.create_load_from_cosphi(net_cigre_mv, buses[5], 0.565, 0.97, "underexcited", name='Load R6')
    pp.create_load_from_cosphi(net_cigre_mv, buses[7], 0.605, 0.97, "underexcited", name='Load R8')
    pp.create_load_from_cosphi(net_cigre_mv, buses[9], 0.490, 0.97, "underexcited", name='Load R10')
    pp.create_load_from_cosphi(net_cigre_mv, buses[10], 0.340, 0.97, "underexcited", name='Load R11')
    # pp.create_load_from_cosphi(net_cigre_mv, buses[11], 15.3, 0.98, "underexcited", name='Load R12')
    pp.create_load_from_cosphi(net_cigre_mv, buses[13], 0.215, 0.97, "underexcited", name='Load R14')

    # Commercial / Industrial
    # pp.create_load_from_cosphi(net_cigre_mv, buses[0], 5.1, 0.95, "underexcited", name='Load CI1')
    pp.create_load_from_cosphi(net_cigre_mv, buses[2], 0.265, 0.85, "underexcited", name='Load CI3')
    pp.create_load_from_cosphi(net_cigre_mv, buses[6], 0.090, 0.85, "underexcited", name='Load CI7')
    pp.create_load_from_cosphi(net_cigre_mv, buses[8], 0.675, 0.85, "underexcited", name='Load CI9')
    pp.create_load_from_cosphi(net_cigre_mv, buses[9], 0.080, 0.85, "underexcited", name='Load CI10')
    # pp.create_load_from_cosphi(net_cigre_mv, buses[11], 5.28, 0.95, "underexcited", name='Load CI12')
    pp.create_load_from_cosphi(net_cigre_mv, buses[12], 0.04, 0.85, "underexcited", name='Load CI13')
    pp.create_load_from_cosphi(net_cigre_mv, buses[13], 0.390, 0.85, "underexcited", name='Load CI14')

    # Optional distributed energy recources
    if with_der in ["pv_wind", "all"]:
        pp.create_sgen(net_cigre_mv, buses[2], 0.02, q_mvar=0, sn_mva=0.02, name='PV 3', type='PV')
        pp.create_sgen(net_cigre_mv, buses[3], 0.020, q_mvar=0, sn_mva=0.02, name='PV 4', type='PV')
        pp.create_sgen(net_cigre_mv, buses[4], 0.030, q_mvar=0, sn_mva=0.03, name='PV 5', type='PV')
        pp.create_sgen(net_cigre_mv, buses[5], 0.030, q_mvar=0, sn_mva=0.03, name='PV 6', type='PV')
        pp.create_sgen(net_cigre_mv, buses[7], 0.030, q_mvar=0, sn_mva=0.03, name='PV 8', type='PV')
        pp.create_sgen(net_cigre_mv, buses[8], 0.030, q_mvar=0, sn_mva=0.03, name='PV 9', type='PV')
        pp.create_sgen(net_cigre_mv, buses[9], 0.040, q_mvar=0, sn_mva=0.04, name='PV 10', type='PV')
        pp.create_sgen(net_cigre_mv, buses[10], 0.010, q_mvar=0, sn_mva=0.01, name='PV 11', type='PV')
        pp.create_sgen(net_cigre_mv, buses[6], 1.5, q_mvar=0, sn_mva=1.5, name='WKA 7',
                       type='WP')
        if with_der == "all":
            pp.create_storage(net_cigre_mv, bus=buses[4], p_mw=0.6, max_e_mwh=nan, sn_mva=0.2,
                              name='Battery 1', type='Battery', max_p_mw=0.6, min_p_mw=-0.6)
            pp.create_sgen(net_cigre_mv, bus=buses[4], p_mw=0.033, sn_mva=0.033,
                           name='Residential fuel cell 1', type='Residential fuel cell')
            pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.310, sn_mva=0.31, name='CHP diesel 1',
                           type='CHP diesel')
            pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.212, sn_mva=0.212, name='Fuel cell 1',
                           type='Fuel cell')
            pp.create_storage(net_cigre_mv, bus=buses[9], p_mw=0.200, max_e_mwh=nan, sn_mva=0.2,
                              name='Battery 2', type='Battery', max_p_mw=0.2, min_p_mw=-0.2)
            pp.create_sgen(net_cigre_mv, bus=buses[9], p_mw=0.014, sn_mva=.014,
                           name='Residential fuel cell 2', type='Residential fuel cell')

    # Bus geo data
    net_cigre_mv.bus_geodata = read_json(io.StringIO(
        """{"x":{"0":7.0,"1":4.0,"2":4.0,"3":4.0,"4":2.5,"5":1.0,"6":1.0,"7":8.0,"8":8.0,"9":6.0,
        "10":4.0,"11":4.0,"12":10.0,"13":10.0,"14":10.0},
        "y":{"0":16,"1":15,"2":13,"3":11,"4":9,
        "5":7,"6":3,"7":3,"8":5,"9":5,"10":5,"11":7,"12":15,"13":11,"14":5},
        "coords":{"0":NaN,"1":NaN,"2":NaN,"3":NaN,"4":NaN,"5":NaN,"6":NaN,"7":NaN,"8":NaN,
        "9":NaN,"10":NaN,"11":NaN,"12":NaN,"13":NaN,"14":NaN}}"""))
    # Match bus.index
    net_cigre_mv.bus_geodata = net_cigre_mv.bus_geodata.loc[net_cigre_mv.bus.index]
    return net_cigre_mv
