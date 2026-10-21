from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.gradio_app import create_demo_ui, get_preloaded_data


@pytest.fixture
def client():
    return TestClient(app)


def test_fastapi_route_endpoint(client):
    response = client.post("/route", json={"task_description": "Optimize matrix multiplication algorithm with cache tiling"})
    assert response.status_code == 200
    data = response.json()
    assert "selected_agent" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["selected_agent"] in ["algorithm", "gpu"]


def test_fastapi_experiment_lifecycle(client):
    # 1. Create experiment
    res_create = client.post("/experiments", json={"task_id": "matrix_multiply", "name": "API Test Exp", "max_rounds": 1})
    assert res_create.status_code == 200
    exp_id = res_create.json()["experiment_id"]

    # 2. Replay endpoint
    res_replay = client.post(f"/experiments/{exp_id}/replay", json={"workers": 2})
    assert res_replay.status_code in [200, 400]  # 400 if pool is empty, 200 if pool exists

    # 3. Policy improve endpoint
    res_improve = client.post(f"/experiments/{exp_id}/improve", json={"round_index": 1, "current_policy_code": ""})
    assert res_improve.status_code == 200
    data_improve = res_improve.json()
    assert "source_code" in data_improve
    assert "version" in data_improve


def test_gradio_app_initialization_and_preloaded_data():
    route_md, df_nodes, ascii_tree, df_replay, df_timeline, metrics_md = get_preloaded_data()
    assert "Laya Routing" in route_md
    assert not df_nodes.empty
    assert "Root:" in ascii_tree or "root" in ascii_tree
    assert not df_replay.empty
    assert not df_timeline.empty
    assert "Best Discovered Quality" in metrics_md

    # Create UI object
    demo = create_demo_ui()
    assert demo is not None

