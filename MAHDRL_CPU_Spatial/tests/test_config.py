from config import SimulationConfig, AgentConfig

def test_paper_baseline_parameters():
    cfg = SimulationConfig()
    assert cfg.area_width_m == 300
    assert cfg.carrier_frequency_hz == 5e9
    assert cfg.bandwidth_hz == 40e6
    assert cfg.max_bs_power_dbm == 30
    assert cfg.ris_elements == 20
    assert cfg.time_slots_per_episode == 100
    assert cfg.discount_factor == 0.95
    assert cfg.soft_update_tau == 0.01
    assert cfg.batch_size == 128
    assert cfg.learning_rate == 0.001
