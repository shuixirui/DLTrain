"""Describe MCore-owned parameter ranges without assuming a full FP32 master copy."""


def owned_parameter_shards(named_params, optimizer):
    names = {id(param): name for name, param in named_params.items()}
    shards, seen = [], set()
    for leaf in getattr(optimizer, "chained_optimizers", [optimizer]):
        assert hasattr(leaf, "_get_model_param_range_map"), "Expected MCore DistributedOptimizer"
        assert not leaf.config.use_precision_aware_optimizer, "Probe requires explicit FP32 masters"
        base = leaf.optimizer
        if hasattr(base, "adam_w_mode"):
            assert base.adam_w_mode, "Expected decoupled AdamW weight decay"
        owners = {id(master): group for group in base.param_groups for master in group["params"]}
        model_groups = leaf.model_float16_groups
        master_groups = leaf.shard_fp32_from_float16_groups
        assert len(model_groups) == len(master_groups)
        assert not any(leaf.model_fp32_groups), "Expected all model parameters to be BF16"
        for model_group, master_group in zip(model_groups, master_groups):
            assert len(model_group) == len(master_group)
            for param, master in zip(model_group, master_group):
                assert master is not None and id(master) in owners
                assert id(param) in names and id(param) not in seen
                seen.add(id(param))
                owned = leaf._get_model_param_range_map(param)["param"]
                start, end = int(owned.start), int(owned.end)
                assert 0 <= start < end <= param.numel(), (names[id(param)], start, end)
                assert end - start == master.numel(), "Master size must match the owned model range"
                assert getattr(param, "main_param", None) is master
                shards.append({
                    "name": names[id(param)], "param": param, "master": master,
                    "base": base, "group": owners[id(master)], "leaf": leaf,
                    "param_start": start, "param_end": end,
                })
    assert shards, "No local optimizer shards"
    return shards


def choose_probe_shards(shards, small_probes):
    requested = {id(param) for _, param in small_probes}
    suffixes = (
        "language_model.decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight0",
        "language_model.decoder.layers.3.self_attention.linear_qkv.weight",
        "language_model.decoder.layers.3.self_attention.linear_proj.weight",
        "language_model.output_layer.weight",
    )
    chosen, seen = [], set()
    # Some dense DP ranks own none of the originally requested parameters.
    # Audit two owned matrices per leaf as well, so each optimizer is checked.
    fallback_counts = {}
    for shard in shards:
        param = shard["param"]
        key = id(shard["leaf"])
        fallback = param.ndim == 2 and "language_model" in shard["name"] and fallback_counts.get(key, 0) < 2
        if fallback:
            fallback_counts[key] = fallback_counts.get(key, 0) + 1
        if id(param) in requested or shard["name"].endswith(suffixes) or fallback:
            if id(param) not in seen:
                seen.add(id(param))
                chosen.append(shard)
    assert chosen, "No owned probe parameters"
    assert {id(shard["leaf"]) for shard in chosen} == {id(shard["leaf"]) for shard in shards}, "Every local optimizer must have a probe"
    return chosen
