import sys
import types

import torch

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForSequenceClassification = object
transformers_stub.Trainer = object
transformers_stub.TrainingArguments = object
transformers_stub.DataCollatorWithPadding = object
sys.modules.setdefault("transformers", transformers_stub)

peft_stub = types.ModuleType("peft")
peft_stub.LoraConfig = object
peft_stub.TaskType = types.SimpleNamespace(SEQ_CLS="SEQ_CLS")
peft_stub.get_peft_model = lambda model, config: model
sys.modules.setdefault("peft", peft_stub)

from client import Client
from server import Server


class FakeModel:
    def __init__(self, params):
        self.params = params

    def named_parameters(self):
        return list(self.params.items())


def make_client(mask=None):
    client = Client(
        client_id=0,
        task_name="debug",
        tokenizer=None,
        model_name="debug",
        num_clients=1,
        idx=0,
        rank_dropout_config={"lambda_e": 1.0, "lambda_b": 1.0, "lambda_a": 1.0},
    )
    client.local_model = FakeModel(
        {
            "layer.lora_A0.weight": torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
            "layer.lora_B0.weight": torch.nn.Parameter(torch.tensor([[5.0, 6.0], [7.0, 8.0]])),
            "layer.lora_svd_e0": torch.nn.Parameter(torch.tensor([0.2, 0.9])),
            "layer.lora_route.weight": torch.nn.Parameter(torch.tensor([[1.0, 2.0]])),
        }
    )
    client.set_active_rank_masks(mask or {})
    return client


def test_sparse_upload_omits_inactive_ranks():
    client = make_client({"layer||0": [True, False]})
    payload = client.get_sparse_lora_params_for_pruning()

    uploaded_ranks = {item["rank_id"] for item in payload["rank_updates"]}
    assert uploaded_ranks == {0}
    assert payload["communication_after"] < payload["communication_before"]


def sparse_update(layer, expert_id, rank_id, part, param_name, value, full_shape):
    return {
        "layer": layer,
        "expert_id": expert_id,
        "rank_id": rank_id,
        "part": part,
        "param_name": param_name,
        "value": torch.tensor(value, dtype=torch.float32),
        "full_shape": full_shape,
    }


def make_payload(client_id, rank0_value, include_rank1=True, rank0_importance=0.1, rank1_importance=0.9):
    updates = [
        sparse_update("layer", 0, 0, "A", "layer.lora_A0.weight", [rank0_value, rank0_value], (2, 2)),
        sparse_update("layer", 0, 0, "B", "layer.lora_B0.weight", [rank0_value, rank0_value], (2, 2)),
        sparse_update("layer", 0, 0, "e", "layer.lora_svd_e0", rank0_value, (2,)),
    ]
    if include_rank1:
        updates.extend([
            sparse_update("layer", 0, 1, "A", "layer.lora_A0.weight", [10.0, 10.0], (2, 2)),
            sparse_update("layer", 0, 1, "B", "layer.lora_B0.weight", [10.0, 10.0], (2, 2)),
            sparse_update("layer", 0, 1, "e", "layer.lora_svd_e0", 10.0, (2,)),
        ])

    return {
        "__sparse_rank_payload__": True,
        "client_id": client_id,
        "dense_params": {},
        "rank_updates": updates,
        "rank_importance": [
            {"layer": "layer", "expert_id": 0, "rank_id": 0, "value": rank0_importance},
            {"layer": "layer", "expert_id": 0, "rank_id": 1, "value": rank1_importance},
        ],
        "communication_before": 10,
        "communication_after": len(updates),
    }


def test_sparse_aggregation_and_missing_rank_preservation():
    server = Server(2, device="cpu", pruning_config={"enable_rank_pruning": True, "pruning_threshold": 0.2, "pruning_patience": 2, "r_min": 1})
    server.server_params["layer.lora_A0.weight"] = torch.tensor([[0.0, 0.0], [9.0, 9.0]])
    server.server_params["layer.lora_B0.weight"] = torch.tensor([[0.0, 9.0], [0.0, 9.0]])
    server.server_params["layer.lora_svd_e0"] = torch.tensor([0.0, 9.0])

    payloads = [make_payload(0, 1.0, include_rank1=False), make_payload(1, 3.0, include_rank1=False)]
    aggregated = server.aggregation_sparse_rank_pruning(
        route_aggregation=False,
        payloads=payloads,
        lora_client_map={0: [0, 1]},
        pruning_enabled_this_round=False,
    )

    assert torch.allclose(aggregated[0]["layer.lora_A0.weight"][0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(aggregated[0]["layer.lora_A0.weight"][1], torch.tensor([9.0, 9.0]))
    assert torch.allclose(aggregated[0]["layer.lora_svd_e0"], torch.tensor([2.0, 9.0]))


def test_ema_patience_and_r_min_pruning():
    server = Server(1, device="cpu", pruning_config={"enable_rank_pruning": True, "pruning_threshold": 0.2, "pruning_patience": 2, "r_min": 1})
    payload = make_payload(0, 1.0, include_rank1=True, rank0_importance=0.1, rank1_importance=0.9)

    server.aggregation_sparse_rank_pruning(False, [payload], {0: [0]}, pruning_enabled_this_round=True)
    assert server.get_active_rank_masks()["layer||0"] == [True, True]
    assert server.get_last_pruning_stats()["newly_pruned"] == 0

    server.aggregation_sparse_rank_pruning(False, [payload], {0: [0]}, pruning_enabled_this_round=True)
    assert server.get_active_rank_masks()["layer||0"] == [False, True]
    assert server.get_last_pruning_stats()["newly_pruned"] == 1

    server.aggregation_sparse_rank_pruning(False, [payload], {0: [0]}, pruning_enabled_this_round=True)
    assert server.get_active_rank_masks()["layer||0"] == [False, True]


def main():
    test_sparse_upload_omits_inactive_ranks()
    test_sparse_aggregation_and_missing_rank_preservation()
    test_ema_patience_and_r_min_pruning()
    print("Rank pruning debug checks passed.")


if __name__ == "__main__":
    main()
