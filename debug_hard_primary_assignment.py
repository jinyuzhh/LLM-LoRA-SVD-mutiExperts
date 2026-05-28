import sys
import types

import numpy as np
import torch

datasets_stub = types.ModuleType("datasets")
datasets_stub.load_dataset = None
datasets_stub.DatasetDict = dict
sys.modules.setdefault("datasets", datasets_stub)

from server import Server
from utils import apply_hard_primary_assignment, select_expert_medoids


def assert_each_client_once(lora_client_map, expected_clients):
    assigned = []
    for clients in lora_client_map.values():
        assigned.extend(clients)
    assert sorted(assigned) == sorted(expected_clients)
    assert len(assigned) == len(set(assigned))


def test_medoid_selection():
    similarity = np.array(
        [
            [1.0, 0.8, 0.2, 0.1, 0.1],
            [0.8, 1.0, 0.9, 0.1, 0.1],
            [0.2, 0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 0.1, 1.0, 0.6],
            [0.1, 0.1, 0.1, 0.6, 1.0],
        ],
        dtype=np.float32,
    )
    medoids = select_expert_medoids(similarity, {0: [0, 1, 2], 1: [3, 4]})
    assert medoids == {0: 1, 1: 3}


def test_margin_switching():
    similarity = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.95, 0.1],
            [0.1, 0.95, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    original_map = {0: [0, 1], 1: [2, 3]}

    no_switch_map, _, no_switch_details, no_switch_count = apply_hard_primary_assignment(
        similarity,
        original_map,
        assignment_margin_delta=0.1,
    )
    assert no_switch_count == 0
    assert no_switch_details[1]["final_expert"] == 0
    assert_each_client_once(no_switch_map, [0, 1, 2, 3])

    switch_map, _, switch_details, switch_count = apply_hard_primary_assignment(
        similarity,
        original_map,
        assignment_margin_delta=0.01,
    )
    assert switch_count == 1
    assert switch_details[1]["final_expert"] == 1
    assert switch_map == {0: [0], 1: [1, 2, 3]}
    assert_each_client_once(switch_map, [0, 1, 2, 3])


def test_server_aggregates_svd_e_by_expert():
    server = Server(clients_num=4, device="cpu")
    params = []
    for value in [1.0, 3.0, 100.0, 200.0]:
        params.append(
            {
                "layer.lora_svd_e0": torch.tensor([value]),
                "layer.lora_svd_e1": torch.tensor([value + 10.0]),
            }
        )

    aggregated = server.aggregation(
        route_aggregation=False,
        params=params,
        lora_client_map={0: [0, 1], 1: [2, 3]},
    )

    assert torch.allclose(aggregated[0]["layer.lora_svd_e0"], torch.tensor([2.0]))
    assert torch.allclose(aggregated[0]["layer.lora_svd_e1"], torch.tensor([160.0]))
    assert torch.allclose(aggregated[3]["layer.lora_svd_e0"], torch.tensor([2.0]))
    assert torch.allclose(aggregated[3]["layer.lora_svd_e1"], torch.tensor([160.0]))


def main():
    test_medoid_selection()
    test_margin_switching()
    test_server_aggregates_svd_e_by_expert()
    print("Hard-primary assignment debug checks passed.")


if __name__ == "__main__":
    main()
