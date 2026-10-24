from scripts.sewer_ml_pipeline.scheduler.cloud_operations import CloudOperations, OperationError


def operations():
    # Reconciliation and validation are pure; avoid acquiring ADC in unit tests.
    return object.__new__(CloudOperations)


def test_stored_running_state_is_not_live_without_resource_id():
    inventory = {"verified_at": "2026-10-23T20:00:00+00:00", "builds": [], "batch_jobs": []}
    result = operations().reconcile_pipeline(
        {"state": "PREPARING", "steps": [{"id": "prepare", "state": "RUNNING"}]}, inventory
    )
    assert result["state"] == "UNVERIFIED"
    assert result["steps"][0]["state"] == "UNVERIFIED"
    assert result["steps"][0]["reported_state"] == "RUNNING"


def test_resource_id_reconciles_against_live_cloud_build():
    inventory = {
        "verified_at": "2026-10-23T20:00:00+00:00",
        "builds": [{"id": "build-123", "status": "WORKING"}],
        "batch_jobs": [],
    }
    result = operations().reconcile_pipeline(
        {"state": "QUEUED", "steps": [{"id": "prepare", "state": "QUEUED", "resource_id": "build-123"}]},
        inventory,
    )
    assert result["state"] == "RUNNING"
    assert result["steps"][0]["state"] == "WORKING"
    assert result["steps"][0]["status_source"] == "LIVE_GCP"


def test_provider_identifiers_reject_parent_traversal():
    try:
        operations()._safe("../secret")
    except OperationError:
        pass
    else:
        raise AssertionError("parent traversal must be rejected")
