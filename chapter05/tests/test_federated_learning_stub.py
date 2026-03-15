"""
test_federated_learning_stub.py
Chapter 5 — Future Frontiers by Jigar Sheth
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from federated_learning_stub import (
    generate_multi_site_dataset, federated_average,
    CrossSiloClinicalFL, CrossCompanyFL, PatientDeviceFL,
    RegulatorInTheLoopFL, FederatedPharmacovigilanceFL
)


def test_multi_site_dataset_correct_shape():
    sites = generate_multi_site_dataset(n_sites=4, n_per_site=100)
    assert len(sites) == 4
    for sid, X, y, n in sites:
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        assert n == 100


def test_fedavg_weights_proportional():
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([3.0, 4.0, 5.0])
    # 100 vs 100 samples — should average equally
    result = federated_average([w1, w2], [100, 100])
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_fedavg_larger_site_dominates():
    w1 = np.array([0.0, 0.0])
    w2 = np.array([10.0, 10.0])
    result = federated_average([w1, w2], [900, 100])
    # Site 1 contributes 90%; result should be close to w1
    assert result[0] < 2.0


def test_cross_silo_training_completes():
    sites = generate_multi_site_dataset(n_sites=3, n_per_site=200)
    fl = CrossSiloClinicalFL(sites, n_rounds=3)
    result = fl.train()
    assert result["rounds_completed"] == 3
    assert 0 < result["final_avg_auc"] < 1.0


def test_cross_company_governance_checklist():
    fl = CrossCompanyFL(["Pfizer", "Roche"])
    checklist = fl.governance_checklist()
    assert len(checklist["governance_items"]) >= 5
    assert "SubstraFL" in checklist["framework"]


def test_patient_device_no_raw_data_transmitted():
    fl = PatientDeviceFL(n_devices=10, epsilon=1.0)
    updates = [fl.simulate_device_update(f"dev_{i}", 50) for i in range(5)]
    for u in updates:
        assert u["raw_data_transmitted"] is False
    agg = fl.aggregate_device_updates(updates)
    assert agg["n_devices_contributed"] == 5


def test_regulator_report_contains_no_patient_data():
    rl = RegulatorInTheLoopFL("FDA CDER", ["Site_A", "Site_B"])
    report = rl.generate_regulatory_report(3, {"auc": 0.81})
    assert "patient" not in str(report).lower() or "no patient" in str(report).lower()
    assert report["data_transmitted"].startswith("Aggregate")
