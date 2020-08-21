#!/usr/bin/env python3
'''
==============================================================
 Copyright © 2019 Intel Corporation

 SPDX-License-Identifier: MIT
==============================================================
'''

import intel.tca as tca

target = tca.get_target(id="whl_u_cnp_lp")
components = [(c.component, tca.latest(c.steppings))
              for c in target.components]
component_config = tca.ComponentWithSelectedSteppingList()
for comp in components:
    config_tmp = tca.ComponentWithSelectedStepping()
    config_tmp.component, config_tmp.stepping = comp
supported_connections = target.get_supported_connection_configurations(
    component_config)


def conn_filter(conn: tca.ConnectionConfiguration) -> bool:
    if conn.type != tca.ConnectionType_IPC:
        return False
    if "CCA" not in conn.ipc_configuration.selection:
        return False
    return True


connection_config = next(filter(conn_filter, supported_connections))
profile = tca.Profile()
profile.name = "My TCA profile"
profile.target = target
profile.component_configuration = component_config
profile.connection_configuration = connection_config
tca.load(profile)
tca.connect()
