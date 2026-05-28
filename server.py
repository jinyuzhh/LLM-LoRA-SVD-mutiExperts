import json
import os
import torch
from typing import List, Dict


class Server:
    def __init__(self, clients_num: int, device: str = "cuda", pruning_config=None):
        self.clients_num = clients_num
        self.device = device
        self.lora_client_map = None  
        self.pruning_config = pruning_config or {}
        self.ema_beta = 0.9
        self.server_params = {}
        self.active_rank_masks = {}
        self.ema_importance = {}
        self.pruning_patience_counters = {}
        self.last_pruning_stats = {
            "communication_before": 0,
            "communication_after": 0,
            "newly_pruned": 0,
            "active_rank_counts": {},
        }

    def get_active_rank_masks(self):
        return {key: list(value) for key, value in self.active_rank_masks.items()}

    def get_last_pruning_stats(self):
        return self.last_pruning_stats

    def _mask_key(self, layer, expert_id):
        return f"{layer}||{expert_id}"

    def _rank_key(self, layer, expert_id, rank_id):
        return f"{layer}||{expert_id}||{rank_id}"

    def _ensure_param(self, param_name, shape, dtype, device):
        if param_name not in self.server_params:
            self.server_params[param_name] = torch.zeros(shape, dtype=dtype, device=device)
        return self.server_params[param_name]

    @staticmethod
    def _write_rank_slice(full_tensor, part, rank_id, value):
        if part == "A":
            full_tensor[rank_id, :] = value
        elif part == "B":
            full_tensor[:, rank_id] = value
        elif part == "e":
            full_tensor[rank_id] = value
        else:
            raise ValueError(f"Unsupported sparse rank part: {part}")

    def _aggregate_dense_sparse_params(self, payloads, client_to_group, route_aggregation):
        dense_results = [{} for _ in payloads]
        all_param_names = sorted({
            param_name
            for payload in payloads
            for param_name in payload.get("dense_params", {}).keys()
        })

        for client_idx, payload in enumerate(payloads):
            for param_name in all_param_names:
                if "lora_route" in param_name and route_aggregation:
                    client_group = client_to_group.get(client_idx)
                    group_indices = self.lora_client_map.get(client_group, []) if client_group is not None else []
                    tensors = [
                        payloads[i]["dense_params"][param_name].to(self.device)
                        for i in group_indices
                        if i < len(payloads) and param_name in payloads[i].get("dense_params", {})
                    ]
                    if tensors:
                        dense_results[client_idx][param_name] = torch.stack(tensors).mean(dim=0)
                    elif param_name in payload.get("dense_params", {}):
                        dense_results[client_idx][param_name] = payload["dense_params"][param_name].to(self.device)
                elif param_name in payload.get("dense_params", {}):
                    dense_results[client_idx][param_name] = payload["dense_params"][param_name].to(self.device)

        return dense_results

    def _update_pruning_importance(self, payloads, client_to_group=None):
        client_to_group = client_to_group or {}
        for client_idx, payload in enumerate(payloads):
            for item in payload.get("rank_importance", []):
                assigned_group = client_to_group.get(client_idx)
                if assigned_group is not None and int(assigned_group) != int(item["expert_id"]):
                    continue

                rank_key = self._rank_key(item["layer"], item["expert_id"], item["rank_id"])
                value = float(item["value"])
                if rank_key in self.ema_importance:
                    self.ema_importance[rank_key] = self.ema_beta * self.ema_importance[rank_key] + (1 - self.ema_beta) * value
                else:
                    self.ema_importance[rank_key] = value

    def _prune_ranks(self, pruning_enabled_this_round):
        newly_pruned = 0

        if not pruning_enabled_this_round:
            return newly_pruned

        threshold = self.pruning_config.get("pruning_threshold", 0.0)
        patience = self.pruning_config.get("pruning_patience", 1)
        r_min = self.pruning_config.get("r_min", 1)

        for mask_key, mask in list(self.active_rank_masks.items()):
            active_indices = [idx for idx, active in enumerate(mask) if active]
            if len(active_indices) <= r_min:
                continue

            layer, expert_id_text = mask_key.rsplit("||", 1)
            expert_id = int(expert_id_text)
            candidates = []

            for rank_id in active_indices:
                rank_key = self._rank_key(layer, expert_id, rank_id)
                ema_value = self.ema_importance.get(rank_key)
                if ema_value is None:
                    continue

                if ema_value < threshold:
                    self.pruning_patience_counters[rank_key] = self.pruning_patience_counters.get(rank_key, 0) + 1
                else:
                    self.pruning_patience_counters[rank_key] = 0

                if self.pruning_patience_counters[rank_key] >= patience:
                    candidates.append((ema_value, rank_id))

            if not candidates:
                continue

            keep_by_min = sorted(
                active_indices,
                key=lambda rank_id: self.ema_importance.get(self._rank_key(layer, expert_id, rank_id), float("inf")),
                reverse=True,
            )[:r_min]
            protected = set(keep_by_min)

            for _, rank_id in sorted(candidates):
                if len([idx for idx, active in enumerate(mask) if active]) <= r_min:
                    break
                if rank_id in protected:
                    continue
                if mask[rank_id]:
                    mask[rank_id] = False
                    newly_pruned += 1

        return newly_pruned

    def _active_rank_counts(self):
        return {key: int(sum(value)) for key, value in self.active_rank_masks.items()}

    def save_pruning_state(self, output_dir, round_idx):
        os.makedirs(output_dir, exist_ok=True)
        state = {
            "active_rank_masks": self.get_active_rank_masks(),
            "ema_importance": self.ema_importance,
            "pruning_patience_counters": self.pruning_patience_counters,
            "last_pruning_stats": self.last_pruning_stats,
        }
        with open(os.path.join(output_dir, f"rank_pruning_state_round_{round_idx}.json"), "w") as f:
            json.dump(state, f, indent=2)

    def aggregation_sparse_rank_pruning(
        self,
        route_aggregation: bool,
        payloads: List[Dict],
        lora_client_map=None,
        pruning_enabled_this_round=False
    ) -> List[Dict]:
        if lora_client_map is not None:
            self.lora_client_map = lora_client_map

        if self.lora_client_map is None:
            raise ValueError("lora_client_map must be provided for sparse rank pruning aggregation")

        client_to_group = {}
        for group_idx, clients in self.lora_client_map.items():
            for client in clients:
                client_to_group[client] = group_idx

        dense_results = self._aggregate_dense_sparse_params(payloads, client_to_group, route_aggregation)
        update_groups = {}

        for client_idx, payload in enumerate(payloads):
            for update in payload.get("rank_updates", []):
                assigned_group = client_to_group.get(client_idx)
                if assigned_group is not None and int(assigned_group) != int(update["expert_id"]):
                    continue

                key = (update["param_name"], update["part"], update["rank_id"])
                update_groups.setdefault(key, []).append(update)

                mask_key = self._mask_key(update["layer"], update["expert_id"])
                if mask_key not in self.active_rank_masks:
                    rank_count = update["full_shape"][0] if update["part"] in {"A", "e"} else update["full_shape"][1]
                    self.active_rank_masks[mask_key] = [True] * rank_count

        for (param_name, part, rank_id), updates in update_groups.items():
            values = [item["value"].to(self.device) for item in updates]
            avg_value = torch.stack(values).mean(dim=0)
            first_update = updates[0]
            full_tensor = self._ensure_param(param_name, first_update["full_shape"], avg_value.dtype, avg_value.device)
            self._write_rank_slice(full_tensor, part, rank_id, avg_value)

        self._update_pruning_importance(payloads, client_to_group)
        newly_pruned = self._prune_ranks(pruning_enabled_this_round)

        communication_before = sum(payload.get("communication_before", 0) for payload in payloads)
        communication_after = sum(payload.get("communication_after", 0) for payload in payloads)
        self.last_pruning_stats = {
            "communication_before": communication_before,
            "communication_after": communication_after,
            "newly_pruned": newly_pruned,
            "active_rank_counts": self._active_rank_counts(),
        }

        aggregated_results = []
        for dense_result in dense_results:
            result = {name: value.clone() for name, value in self.server_params.items()}
            result.update(dense_result)
            aggregated_results.append(result)

        return aggregated_results

    def aggregation_warmup(self, route_aggregation: bool, params: List, lora_client_map=None) -> List[Dict]:
        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]

        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]

        final_warmup_round = lora_client_map is not None

        if final_warmup_round:
            self.lora_client_map = lora_client_map
            print("Final warmup round, preparing transition to clustered LoRA")

            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    aggregated_results[client_idx][param_name] = param_value

            client_to_group = {}
            for group_idx, clients in lora_client_map.items():
                for client in clients:
                    client_to_group[client] = int(group_idx)

            for group_idx, group_clients in lora_client_map.items():
                group_idx = int(group_idx)

                if not group_clients:
                    continue

                print(f"Processing group {group_idx} with clients {group_clients}")

                valid_clients = [c for c in group_clients if c < num_clients]

                if not valid_clients:
                    continue

                for base_param_name in list(gpu_params[0].keys()):
                    if 'lora_A0' in base_param_name or 'lora_B0' in base_param_name or 'lora_svd_e0' in base_param_name:
                        target_param_name = base_param_name.replace('0', str(group_idx))

                        try:
                            stacked_params = torch.stack([
                                gpu_params[i][base_param_name]
                                for i in valid_clients if base_param_name in gpu_params[i]
                            ]).to(self.device)

                            if stacked_params.size(0) > 0:
                                avg_param = stacked_params.mean(dim=0)

                                for client_idx in group_clients:
                                    if client_idx < num_clients:
                                        aggregated_results[client_idx][target_param_name] = avg_param
                        except Exception as e:
                            print(f"Error aggregating {base_param_name} for group {group_idx}: {e}")
        else:
            for client_idx in range(num_clients):
                for param_name, param_value in gpu_params[client_idx].items():
                    if 'lora_A' in param_name or 'lora_B' in param_name or 'lora_svd_e' in param_name or 'lora_route' in param_name:
                        aggregated_results[client_idx][param_name] = param_value

        return aggregated_results
    
    def aggregation(self, route_aggregation: bool, params: List, lora_client_map=None) -> List[Dict]:
        if params and isinstance(params[0], dict) and params[0].get("__sparse_rank_payload__"):
            return self.aggregation_sparse_rank_pruning(
                route_aggregation=route_aggregation,
                payloads=params,
                lora_client_map=lora_client_map,
                pruning_enabled_this_round=self.pruning_config.get("pruning_enabled_this_round", False),
            )

        if lora_client_map is not None:
            self.lora_client_map = lora_client_map

        if self.lora_client_map is None:
            raise ValueError("lora_client_map must be provided for aggregation after warmup phase")

        client_to_group = {}
        for group_idx, clients in self.lora_client_map.items():
            for client in clients:
                client_to_group[client] = group_idx

        gpu_params = [
            {k: v.to(self.device) for k, v in client_params.items()}
            for client_params in params
        ]
        num_clients = len(gpu_params)
        aggregated_results = [{} for _ in range(num_clients)]
        param_names = gpu_params[0].keys()

        for client_idx in range(num_clients):
            for param_name in param_names:

                if 'lora_route' in param_name:
                    if route_aggregation:
                        client_group = client_to_group.get(client_idx)
                        if client_group is not None:
                            group_indices = self.lora_client_map[client_group]
                            stacked_params = torch.stack([
                                gpu_params[i][param_name]
                                for i in group_indices
                            ]).to(self.device)
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

                elif 'lora_A' in param_name or 'lora_B' in param_name or 'lora_svd_e' in param_name:
                    if 'lora_A' in param_name:
                        lora_idx = int(param_name.split('lora_A')[1][0])
                    elif 'lora_B' in param_name:
                        lora_idx = int(param_name.split('lora_B')[1][0])
                    else:
                        lora_idx = int(param_name.split('lora_svd_e')[1][0])

                    group_indices = self.lora_client_map.get(str(lora_idx), [])
                    if not group_indices:
                        group_indices = self.lora_client_map.get(lora_idx, [])

                    if group_indices:
                        stacked_params = torch.stack([
                            gpu_params[i][param_name]
                            for i in group_indices if i < len(gpu_params) and param_name in gpu_params[i]
                        ]).to(self.device)
                        if stacked_params.size(0) > 0:
                            aggregated_results[client_idx][param_name] = stacked_params.mean(dim=0)
                        else:
                            aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                    else:
                        aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]
                else:
                    aggregated_results[client_idx][param_name] = gpu_params[client_idx][param_name]

        return aggregated_results


