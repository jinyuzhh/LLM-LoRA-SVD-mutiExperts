import torch

from debug_svd_lora import load_lora_linear


def build_layer(p_base=0.5):
    Linear = load_lora_linear()
    layer = Linear(
        3,
        2,
        r=2,
        lora_alpha=2,
        lora_nums=1,
        lora_dropout=0.0,
        adaptive=False,
        k=1,
        bias=False,
        adapter_type="svd_lora",
        enable_rank_dropout=True,
        p_base=p_base,
        lambda_e=1.0,
        lambda_b=1.0,
        lambda_a=1.0,
    )
    return layer


def test_rank_importance_formula():
    layer = build_layer()

    with torch.no_grad():
        layer.lora_svd_e0.copy_(torch.tensor([2.0, -4.0]))
        layer.lora_B0.weight.copy_(torch.tensor([[1.0, -3.0], [5.0, 7.0]]))
        layer.lora_A0.weight.copy_(torch.tensor([[1.0, -1.0, 3.0], [2.0, -4.0, 6.0]]))

    e_score = torch.tensor([2.0, 4.0])
    b_score = torch.tensor([3.0, 5.0])
    a_score = torch.tensor([5.0 / 3.0, 4.0])
    raw = e_score + b_score + a_score
    expected = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    actual = layer._rank_importance(0)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dropout_probability_order_and_whole_rank_mask():
    layer = build_layer(p_base=1.0)
    layer.train()
    layer.set_rank_dropout_gamma(1.0)

    with torch.no_grad():
        layer.lora_svd_e0.copy_(torch.tensor([0.0, 10.0]))
        layer.lora_A0.weight.zero_()
        layer.lora_B0.weight.zero_()

    importance = layer._rank_importance(0)
    dropout_prob = layer.p_base * (1 - importance) * layer.rank_dropout_gamma
    assert dropout_prob[0] > dropout_prob[1]

    hidden = torch.ones(8, 2)
    torch.manual_seed(5)
    dropped = layer._apply_rank_dropout(0, hidden)
    assert torch.allclose(dropped[:, 0], torch.zeros_like(dropped[:, 0]), atol=1e-6)
    assert torch.allclose(dropped[:, 1], torch.ones_like(dropped[:, 1]), atol=1e-6)


def test_eval_and_gamma_zero_disable_dropout():
    layer = build_layer(p_base=1.0)
    hidden = torch.ones(4, 2)

    with torch.no_grad():
        layer.lora_svd_e0.copy_(torch.tensor([0.0, 10.0]))
        layer.lora_A0.weight.zero_()
        layer.lora_B0.weight.zero_()

    layer.eval()
    layer.set_rank_dropout_gamma(1.0)
    eval_out = layer._apply_rank_dropout(0, hidden)
    assert torch.allclose(eval_out, hidden)

    layer.train()
    layer.set_rank_dropout_gamma(0.0)
    gamma_zero_out = layer._apply_rank_dropout(0, hidden)
    assert torch.allclose(gamma_zero_out, hidden)


def test_inverted_scaling_stabilizes_mean():
    layer = build_layer(p_base=0.5)
    layer.train()
    layer.set_rank_dropout_gamma(1.0)

    with torch.no_grad():
        layer.lora_svd_e0.copy_(torch.tensor([1.0, 1.0]))
        layer.lora_A0.weight.zero_()
        layer.lora_B0.weight.zero_()

    hidden = torch.ones(128, 2)
    samples = []
    torch.manual_seed(11)
    for _ in range(2000):
        samples.append(layer._apply_rank_dropout(0, hidden).mean())

    empirical_mean = torch.stack(samples).mean()
    assert torch.allclose(empirical_mean, torch.tensor(1.0), atol=0.05)


def main():
    test_rank_importance_formula()
    test_dropout_probability_order_and_whole_rank_mask()
    test_eval_and_gamma_zero_disable_dropout()
    test_inverted_scaling_stabilizes_mean()
    print("Rank dropout debug checks passed.")


if __name__ == "__main__":
    main()
