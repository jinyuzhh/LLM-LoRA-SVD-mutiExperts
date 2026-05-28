#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import json
import datetime
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import partition_multi_task_dataset, compute_lora_client_map
from client import Client, WarmupClient
from server import Server


def log_rank_dropout_stats(log_file, label, stats):
    if stats is None:
        message = f"{label} rank dropout stats: unavailable"
    else:
        message = (
            f"{label} rank dropout stats: "
            f"avg_dropout_rate={stats['dropout_rate']:.6f}, "
            f"avg_rank_importance={stats['rank_importance']:.6f}"
        )
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")


def average_rank_dropout_stats(stats_list):
    valid_stats = [stats for stats in stats_list if stats is not None]
    if not valid_stats:
        return None
    return {
        "dropout_rate": sum(stats["dropout_rate"] for stats in valid_stats) / len(valid_stats),
        "rank_importance": sum(stats["rank_importance"] for stats in valid_stats) / len(valid_stats),
    }


def pruning_enabled_for_round(pruning_config, rank_dropout_config, assignment_type, round_idx, round_warmup):
    if not pruning_config.get("enable_rank_pruning", False):
        return False
    if assignment_type != "hard_primary":
        return False
    if round_idx < round_warmup:
        return False

    if not rank_dropout_config.get("enable_rank_dropout", False):
        return True

    dropout_stop_round = rank_dropout_config.get("dropout_stop_round", -1)
    return dropout_stop_round >= 0 and round_idx >= dropout_stop_round


def log_pruning_stats(log_file, round_idx, stats):
    message = (
        f"Round {round_idx + 1} pruning stats: "
        f"communication_before={stats.get('communication_before', 0)}, "
        f"communication_after={stats.get('communication_after', 0)}, "
        f"newly_pruned={stats.get('newly_pruned', 0)}"
    )
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")
        f.write("Active rank count per expert/layer:\n")
        for key, count in sorted(stats.get("active_rank_counts", {}).items()):
            f.write(f"  {key}: {count}\n")


def train_federated(
    dummy,
    clients,
    server,
    global_rounds,
    local_epochs,
    output_dir,
    lr=3e-4,
    round_warmup=1,
    max_clusters=10,
    task_info=None,
    client_datasets=None,
    batch_size=128,
    similarity_type="fedlease_original",
    assignment_type="fedlease_top_m",
    assignment_margin_delta=0.0,
    pruning_config=None
):
    pruning_config = pruning_config or {}
    personal_dir = os.path.join(output_dir, "proposed_m2")
    os.makedirs(personal_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(personal_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write(f"[{current_time}] Starting Federated Training with Dummy Client\n")
        f.write(f"Total Rounds: {global_rounds}, Local Epochs: {local_epochs}, Warmup Rounds: {round_warmup}\n")
        if clients:
            f.write(f"Rank Dropout Config: {clients[0].rank_dropout_config}\n")
        f.write(f"Rank Pruning Config: {pruning_config}\n")
        f.write("-" * 50 + "\n")
    
    warmup_clients = clients
    
    all_client_scores = {client.client_id: [] for client in warmup_clients}
    
    lora_client_map = None
    saved_params = None
    optimal_n_clusters = None
    clustered_clients = None
    aggregated_params = None
    
    for round_idx in tqdm(range(global_rounds), desc="Global Rounds"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"\n[{current_time}] Starting Global Round {round_idx + 1}/{global_rounds}\n")
        
        print(f"\nGlobal Round {round_idx + 1}/{global_rounds}")
        
        if round_idx < round_warmup:
            print(f"Running warmup phase (round {round_idx+1}/{round_warmup})")
            
            with open(log_file, "a") as f:
                f.write("Starting dummy client warmup\n")
            
            dummy.load_model()
            round_rank_dropout_stats = []
            dummy_stats = dummy.local_training(
                lr=lr,
                epochs=local_epochs,
                batch_size=batch_size,
                current_round=round_idx,
                is_warmup=True
            )
            log_rank_dropout_stats(log_file, f"Dummy Client {dummy.client_id}", dummy_stats)
            round_rank_dropout_stats.append(dummy_stats)
            dummy.unload_model()
            
            client_params = []
            for client in tqdm(warmup_clients, desc="Client Training (Warmup)"):
                with open(log_file, "a") as f:
                    f.write(f"Training Warmup Client {client.client_id} ({client.task_name})...\n")
                
                client.load_model()
                if round_idx > 0 and aggregated_params is not None:
                    client.load_params(aggregated_params[client.client_id])
                
                client_stats = client.local_training(
                    lr=lr,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    current_round=round_idx,
                    is_warmup=True
                )
                log_rank_dropout_stats(log_file, f"Warmup Client {client.client_id}", client_stats)
                round_rank_dropout_stats.append(client_stats)
                params = client.get_lora_params_and_save_by_module(round_id=round_idx, personal_dir=personal_dir)
                client_params.append(params['params'])
                
                if (round_idx + 1) == round_warmup:
                    if saved_params is None:
                        saved_params = {}
                    saved_params[client.client_id] = params['params']
                
                client.unload_model()

            log_rank_dropout_stats(
                log_file,
                f"Round {round_idx + 1} average",
                average_rank_dropout_stats(round_rank_dropout_stats)
            )
            
            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Warmup)...\n")
            
            agg_lora_client_map = None
            if (round_idx + 1) == round_warmup:
                with open(log_file, "a") as f:
                    f.write("Computing LoRA client mapping for clustering\n")
                
                lora_client_map, optimal_n_clusters = compute_lora_client_map(
                    warmup_clients, 
                    round_idx, 
                    personal_dir,
                    max_clusters=max_clusters,
                    similarity_type=similarity_type,
                    assignment_type=assignment_type,
                    assignment_margin_delta=assignment_margin_delta
                )
                agg_lora_client_map = lora_client_map
                
                with open(log_file, "a") as f:
                    f.write(f"LoRA client mapping computed: {lora_client_map}\n")
                    f.write(f"Optimal number of clusters: {optimal_n_clusters}\n")
                
                if task_info is not None and client_datasets is not None:
                    clustered_clients = []
                    for client_id in range(len(warmup_clients)):
                        client_task = task_info[client_id]["task_name"]
                        num_labels = task_info[client_id]["num_labels"]
                        
                        client_cluster = None
                        for cluster_id, cluster_clients in lora_client_map.items():
                            if client_id in cluster_clients:
                                client_cluster = int(cluster_id)
                                break
                        
                        if client_cluster is None:
                            print(f"Warning: Client {client_id} not found in any cluster. Assigning to cluster 0.")
                            client_cluster = 0
                        
                        client = Client(
                            client_id=client_id,
                            task_name=client_task,
                            tokenizer=warmup_clients[client_id].tokenizer,
                            model_name=warmup_clients[client_id].model_name,
                            num_clients=len(warmup_clients),
                            rank=4,
                            lora_n=optimal_n_clusters,
                            adaptive=True,
                            cache_path=output_dir,
                            idx=client_cluster,
                            rank_dropout_config=warmup_clients[client_id].rank_dropout_config,
                            pruning_config=pruning_config
                        )
                        
                        client.set_dataset(client_datasets[client_id], num_labels)
                        
                        clustered_clients.append(client)
                    
                    clients = clustered_clients
                    
                    server = Server(clients_num=len(clients), pruning_config=pruning_config)
                    
                    with open(log_file, "a") as f:
                        f.write(f"Initialized {len(clients)} clustered clients with {optimal_n_clusters} LoRA modules\n")
            
            aggregated_params = server.aggregation_warmup(
                route_aggregation=True,
                params=client_params,
                lora_client_map=agg_lora_client_map
            )
            
            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing warmup evaluation at round {round_idx + 1}\n")
                
                print(f"\nWarmup Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                
                for client in warmup_clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    
                    metrics = client.evaluate_model()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    
                    client.unload_model()
                
                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx+1}.json")
                with open(summary_file, 'w') as f:
                    json.dump(round_scores, f, indent=2)
        
        else:
            print(f"Running clustered training phase (round {round_idx+1-round_warmup}/{global_rounds-round_warmup})")
            
            if clients is None:
                with open(log_file, "a") as f:
                    f.write("ERROR: Clustered clients not initialized. This should not happen.\n")
                raise RuntimeError("Clustered clients not initialized")
            
            if round_idx == round_warmup:
                with open(log_file, "a") as f:
                    f.write("Transitioning from warmup to clustered training\n")
                    f.write(f"LoRA client mapping: {lora_client_map}\n")
                
                for client in clients:
                    client.load_model()
                    
                    if client.client_id in saved_params:
                        warmed_params = {}
                        client_group = client.idx
                        
                        for name, param in saved_params[client.client_id].items():
                            if 'lora_A0' in name and client_group is not None:
                                new_name = name.replace('lora_A0', f'lora_A{client_group}')
                                warmed_params[new_name] = param
                            elif 'lora_B0' in name and client_group is not None:
                                new_name = name.replace('lora_B0', f'lora_B{client_group}')
                                warmed_params[new_name] = param
                            elif 'lora_svd_e0' in name and client_group is not None:
                                new_name = name.replace('lora_svd_e0', f'lora_svd_e{client_group}')
                                warmed_params[new_name] = param
                            elif 'lora_route' in name:
                                continue
                            else:
                                warmed_params[name] = param
                        
                        client.local_model.load_state_dict(warmed_params, strict=False)
                    
                    client.unload_model()
            
            client_params = []
            round_rank_dropout_stats = []
            use_sparse_pruning_upload = pruning_config.get("enable_rank_pruning", False) and assignment_type == "hard_primary"
            if use_sparse_pruning_upload and clients:
                clients[0].load_model()
                use_sparse_pruning_upload = clients[0].has_svd_lora_params()
                clients[0].unload_model()

            active_rank_masks = server.get_active_rank_masks() if hasattr(server, "get_active_rank_masks") else {}
            for client in tqdm(clients, desc="Client Training (Clustered)"):
                with open(log_file, "a") as f:
                    f.write(f"Training Clustered Client {client.client_id} ({client.task_name})...\n")
                
                client.set_active_rank_masks(active_rank_masks)
                client.load_model()
                if round_idx > round_warmup:
                    client.load_params(aggregated_params[client.client_id])
                
                client_stats = client.local_training(
                    lr=lr,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    lora_client_map=lora_client_map,
                    current_round=round_idx,
                    is_warmup=False
                )
                log_rank_dropout_stats(log_file, f"Clustered Client {client.client_id}", client_stats)
                round_rank_dropout_stats.append(client_stats)
                if use_sparse_pruning_upload:
                    client_params.append(client.get_sparse_lora_params_for_pruning())
                else:
                    params = client.get_lora_params()
                    client_params.append(params['params'])
                
                client.unload_model()

            log_rank_dropout_stats(
                log_file,
                f"Round {round_idx + 1} average",
                average_rank_dropout_stats(round_rank_dropout_stats)
            )
            
            with open(log_file, "a") as f:
                f.write("Starting Server Aggregation (Clustered)...\n")

            if hasattr(server, "pruning_config"):
                server.pruning_config["pruning_enabled_this_round"] = pruning_enabled_for_round(
                    pruning_config,
                    clients[0].rank_dropout_config if clients else {},
                    assignment_type,
                    round_idx,
                    round_warmup
                )
            
            aggregated_params = server.aggregation(
                route_aggregation=True,
                params=client_params,
                lora_client_map=lora_client_map
            )

            if use_sparse_pruning_upload and hasattr(server, "get_active_rank_masks"):
                active_rank_masks = server.get_active_rank_masks()
                for client in clients:
                    client.set_active_rank_masks(active_rank_masks)
                log_pruning_stats(log_file, round_idx, server.get_last_pruning_stats())
                server.save_pruning_state(personal_dir, round_idx)
            
            if (round_idx + 1) % 1 == 0:
                with open(log_file, "a") as f:
                    f.write(f"Performing clustered evaluation at round {round_idx + 1}\n")
                
                print(f"\nClustered Round {round_idx + 1} Evaluation Scores:")
                round_scores = {}
                
                for client in clients:
                    client_id = client.client_id
                    client.load_model()
                    client.load_params(aggregated_params[client_id])
                    
                    metrics = client.evaluate_model()
                    all_client_scores[client_id].append(metrics)
                    round_scores[client_id] = metrics
                    
                    client.unload_model()
                
                summary_file = os.path.join(personal_dir, f"round_summary_{round_idx+1}.json")
                with open(summary_file, 'w') as f:
                    json.dump(round_scores, f, indent=2)
        
        with open(log_file, "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{current_time}] Completed Global Round {round_idx + 1}/{global_rounds}\n")
            f.write("-" * 50 + "\n")
    
    with open(os.path.join(personal_dir, "training_history.json"), 'w') as f:
        json.dump({
            "client_scores": all_client_scores,
            "optimal_n_clusters": optimal_n_clusters,
            "lora_client_map": {str(k): v for k, v in lora_client_map.items()} if lora_client_map else None
        }, f, indent=2)
    
    return all_client_scores


def parse_args():
    parser = argparse.ArgumentParser(description="FedLEASE: Adaptive LoRA Experts Allocation and Selection")
    
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="Pre-trained model name")
    parser.add_argument("--tasks", nargs="+", default=["sst2", "sst2", "sst2", "sst2", 
                                                         "qnli", "qnli", "qnli", "qnli",
                                                         "mrpc", "mrpc", "mrpc", "mrpc",
                                                         "qqp", "qqp", "qqp", "qqp"],
                        help="List of tasks for each client")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--global_rounds", type=int, default=25,
                        help="Number of global federated rounds")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="Number of local training epochs")
    parser.add_argument("--warmup_rounds", type=int, default=5,
                        help="Number of warmup rounds before clustering")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank")
    parser.add_argument("--max_clusters", type=int, default=4,
                        help="Maximum number of LoRA expert clusters")
    parser.add_argument("--train_samples", type=int, default=1000,
                        help="Training samples per client")
    parser.add_argument("--test_samples", type=int, default=200,
                        help="Test samples per client")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--similarity_type", type=str, default="fedlease_original",
                        choices=["fedlease_original", "svd_b_e"],
                        help="Client similarity backend for clustering")
    parser.add_argument("--assignment_type", type=str, default="fedlease_top_m",
                        choices=["fedlease_top_m", "hard_primary"],
                        help="Expert assignment strategy after clustering")
    parser.add_argument("--assignment_margin_delta", type=float, default=0.0,
                        help="Minimum similarity improvement required to switch primary expert")
    parser.add_argument("--enable_rank_dropout", action="store_true",
                        help="Enable rank-wise dropout for SVD-LoRA layers")
    parser.add_argument("--p_base", type=float, default=0.0,
                        help="Base probability for SVD-LoRA rank-wise dropout")
    parser.add_argument("--dropout_warmup_rounds", type=int, default=0,
                        help="Global rounds before rank-wise dropout starts")
    parser.add_argument("--dropout_ramp_rounds", type=int, default=0,
                        help="Global rounds used to ramp rank-wise dropout gamma")
    parser.add_argument("--dropout_stop_round", type=int, default=-1,
                        help="Global round at which rank-wise dropout is disabled; -1 means never")
    parser.add_argument("--lambda_e", type=float, default=1.0,
                        help="Rank importance weight for SVD e")
    parser.add_argument("--lambda_b", type=float, default=1.0,
                        help="Rank importance weight for LoRA B")
    parser.add_argument("--lambda_a", type=float, default=1.0,
                        help="Rank importance weight for LoRA A")
    parser.add_argument("--enable_rank_pruning", action="store_true",
                        help="Enable server-side expert-rank pruning for SVD-LoRA")
    parser.add_argument("--pruning_threshold", type=float, default=0.0,
                        help="EMA importance threshold for rank pruning")
    parser.add_argument("--pruning_patience", type=int, default=1,
                        help="Consecutive below-threshold rounds required before pruning")
    parser.add_argument("--r_min", type=int, default=1,
                        help="Minimum active ranks to keep per expert/layer")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    task_name_list = args.tasks
    client_num = len(task_name_list)
    
    output_dir = os.path.join(args.output_dir, f"{args.model_name.replace('/', '_')}_multi_task_federated_{client_num}")
    rank_dropout_config = {
        "enable_rank_dropout": args.enable_rank_dropout,
        "p_base": args.p_base,
        "dropout_warmup_rounds": args.dropout_warmup_rounds,
        "dropout_ramp_rounds": args.dropout_ramp_rounds,
        "dropout_stop_round": args.dropout_stop_round,
        "lambda_e": args.lambda_e,
        "lambda_b": args.lambda_b,
        "lambda_a": args.lambda_a,
    }
    pruning_config = {
        "enable_rank_pruning": args.enable_rank_pruning,
        "pruning_threshold": args.pruning_threshold,
        "pruning_patience": args.pruning_patience,
        "r_min": args.r_min,
    }
    
    print(f"Running federated learning with multi-task datasets: {task_name_list}")
    print(f"Number of clients: {client_num}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    client_datasets, task_info = partition_multi_task_dataset(
        task_name_list=task_name_list,
        tokenizer=tokenizer,
        alpha=100000,
        train_samples_per_client=args.train_samples,
        test_samples_per_client=args.test_samples,
        seed=args.seed
    )
    
    dummy_task = task_name_list[0]
    dummy_num_labels = task_info[0]["num_labels"]
    
    dummy = WarmupClient(
        client_id=client_num,
        task_name=dummy_task,
        tokenizer=tokenizer,
        model_name=args.model_name,
        num_clients=client_num,
        rank=args.rank,
        cache_path=output_dir,
        rank_dropout_config=rank_dropout_config,
        pruning_config=pruning_config
    )
    dummy.set_dataset(client_datasets[0], dummy_num_labels)
    
    warmup_clients = []
    for client_id in range(client_num):
        client_task = task_info[client_id]["task_name"]
        num_labels = task_info[client_id]["num_labels"]
        
        client = WarmupClient(
            client_id=client_id,
            task_name=client_task,
            tokenizer=tokenizer,
            model_name=args.model_name,
            num_clients=client_num,
            rank=args.rank,
            cache_path=output_dir,
            rank_dropout_config=rank_dropout_config,
            pruning_config=pruning_config
        )
        client.set_dataset(client_datasets[client_id], num_labels)
        warmup_clients.append(client)
    
    warmup_server = Server(clients_num=len(warmup_clients), pruning_config=pruning_config)
    
    train_result = train_federated(
        dummy=dummy,
        clients=warmup_clients,
        server=warmup_server,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        output_dir=output_dir,
        lr=args.lr,
        round_warmup=args.warmup_rounds,
        max_clusters=args.max_clusters,
        task_info=task_info,
        client_datasets=client_datasets,
        batch_size=args.batch_size,
        similarity_type=args.similarity_type,
        assignment_type=args.assignment_type,
        assignment_margin_delta=args.assignment_margin_delta,
        pruning_config=pruning_config
    )
    
    print("\nTraining completed!")
    print("Final Evaluation Scores for each client:", train_result)


if __name__ == "__main__":
    main()
