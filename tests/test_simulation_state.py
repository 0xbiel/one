from datetime import datetime,timezone
from app.simulation_state import simulation_state,SIMULATION_METADATA_KEY

NOW=datetime(2026,10,1,12,tzinfo=timezone.utc)
def camera():return {'id':'cam','home_id':'home','enabled':True,'metadata':{'simulated':True,SIMULATION_METADATA_KEY:{'synthetic':True,'enabled':True,'home_id':'home','camera_id':'cam','map_id':'map','care_recipient_id':'bobby','started_at':'2026-10-01T00:00:00Z','last_simulated_at':'2026-10-01T11:00:00Z','expires_at':'2026-10-02T00:00:00Z'}}}
def test_online_is_explicit_simulation_never_real_capture_or_ready():
    result=simulation_state(camera(),NOW)
    assert result['status']=='simulated_online' and result['label']=='Simulated online'
    assert result['data_origin']=='simulated' and not result['real_capture'] and not result['ready']
def test_cross_home_or_unmarked_camera_never_simulates():
    row=camera();row['home_id']='other';assert simulation_state(row,NOW)['status'] is None
    row=camera();row['metadata']['simulated']=False;assert simulation_state(row,NOW)['status'] is None
def test_expired_future_disabled_and_opt_out_are_simulated_offline():
    row=camera();assert simulation_state(row,datetime(2026,10,2,tzinfo=timezone.utc))['status']=='simulated_offline'
    row['enabled']=False;assert simulation_state(row,NOW)['status']=='simulated_offline'
    row=camera();row['metadata'][SIMULATION_METADATA_KEY]['enabled']=False;assert simulation_state(row,NOW)['status']=='simulated_offline'
    row=camera();row['metadata'][SIMULATION_METADATA_KEY]['last_simulated_at']='2026-10-01T13:00:00Z';assert simulation_state(row,NOW)['status']=='simulated_offline'
def test_unbounded_and_naive_marker_rejected():
    row=camera();row['metadata'][SIMULATION_METADATA_KEY]['expires_at']='2026-10-09T00:00:00Z';assert simulation_state(row,NOW)['status'] is None
    row=camera();row['metadata'][SIMULATION_METADATA_KEY]['started_at']='2026-10-01T00:00:00';assert simulation_state(row,NOW)['status'] is None
