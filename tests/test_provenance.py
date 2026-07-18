"""Target-manifest round trip, drift detection, and dump sanity checks."""
import json

import mlx.core as mx
import numpy as np
import pytest

from alis_dwq import provenance


def fake_split(n_samples, seed, length=32):
    rng = np.random.default_rng(seed)
    return [(rng.integers(1, 1000, size=length).tolist(), 0)
            for _ in range(n_samples)]


def write_fake_targets(target_dir, manifest, constant=False, lead_dim=None):
    rng = np.random.default_rng(0)
    for split, info in manifest["splits"].items():
        d = target_dir / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(info["batches"]):
            shape = (lead_dim or manifest["batch_size"], 8, manifest["top_k"])
            logits = (np.zeros(shape, np.float32) if constant
                      else rng.normal(size=shape).astype(np.float32))
            mx.save_safetensors(
                str(d / f"{i:010d}.safetensors"),
                {"logits": mx.array(logits),
                 "indices": mx.array(np.zeros(shape, np.uint32))})


def make_manifest(train, valid, bs=1, seq=64, seed=7):
    return provenance.build_manifest(train, valid, bs, seq, seed)


def test_token_stream_hash_sensitivity():
    a = fake_split(4, seed=1)
    assert provenance.token_stream_hash(a) == provenance.token_stream_hash(a)
    # order, content, and offsets all feed the replay invariant
    assert provenance.token_stream_hash(a) != \
        provenance.token_stream_hash(list(reversed(a)))
    bumped = [(t, o + 1) for t, o in a]
    assert provenance.token_stream_hash(a) != provenance.token_stream_hash(bumped)


def test_manifest_round_trip(tmp_path):
    train, valid = fake_split(5, 1), fake_split(2, 2)
    m = make_manifest(train, valid)
    provenance.write_manifest(tmp_path, m)
    write_fake_targets(tmp_path, m)
    provenance.sanity_check_targets(tmp_path, m)
    got = provenance.verify_targets_for_training(tmp_path, train, valid, 1, 64, 7)
    assert got["splits"]["train"]["token_sha256"] == \
        m["splits"]["train"]["token_sha256"]


@pytest.mark.parametrize("kw,field", [
    (dict(bs=2), "batch_size"),
    (dict(seq=128), "max_seq_length"),
    (dict(seed=8), "seed"),
])
def test_verify_refuses_param_drift(tmp_path, kw, field):
    train, valid = fake_split(5, 1), fake_split(2, 2)
    provenance.write_manifest(tmp_path, make_manifest(train, valid))
    args = dict(bs=1, seq=64, seed=7)
    args.update(kw)
    with pytest.raises(SystemExit) as e:
        provenance.verify_targets_for_training(
            tmp_path, train, valid, args["bs"], args["seq"], args["seed"])
    assert field in str(e.value)


def test_verify_refuses_data_drift(tmp_path):
    train, valid = fake_split(5, 1), fake_split(2, 2)
    provenance.write_manifest(tmp_path, make_manifest(train, valid))
    # same sizes, different tokens — the silent case batch-count checks miss
    other = fake_split(5, 99)
    with pytest.raises(SystemExit) as e:
        provenance.verify_targets_for_training(tmp_path, other, valid, 1, 64, 7)
    assert "token_sha256" in str(e.value)


def test_verify_refuses_missing_manifest(tmp_path, monkeypatch):
    train, valid = fake_split(5, 1), fake_split(2, 2)
    with pytest.raises(SystemExit):
        provenance.verify_targets_for_training(tmp_path, train, valid, 1, 64, 7)
    monkeypatch.setenv("ALIS_DWQ_ALLOW_UNVERIFIED_TARGETS", "1")
    assert provenance.verify_targets_for_training(
        tmp_path, train, valid, 1, 64, 7) is None


def test_sanity_catches_partial_dump(tmp_path):
    train, valid = fake_split(5, 1), fake_split(2, 2)
    m = make_manifest(train, valid)
    write_fake_targets(tmp_path, m)
    (tmp_path / "train" / f"{4:010d}.safetensors").unlink()
    with pytest.raises(SystemExit) as e:
        provenance.sanity_check_targets(tmp_path, m)
    assert "partial" in str(e.value)


def test_sanity_catches_zeroed_files(tmp_path):
    train, valid = fake_split(2, 1), fake_split(1, 2)
    m = make_manifest(train, valid)
    write_fake_targets(tmp_path, m, constant=True)
    with pytest.raises(SystemExit) as e:
        provenance.sanity_check_targets(tmp_path, m)
    assert "constant" in str(e.value)


def test_sanity_catches_all_gather_lead_dim(tmp_path):
    # the distributed (ranks, seq, k) corruption: leading dim != batch_size
    train, valid = fake_split(2, 1), fake_split(1, 2)
    m = make_manifest(train, valid)
    write_fake_targets(tmp_path, m, lead_dim=2)
    with pytest.raises(SystemExit) as e:
        provenance.sanity_check_targets(tmp_path, m)
    assert "leading dim" in str(e.value)


def test_event_log_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIS_DWQ_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(provenance, "_run_dir", None)
    provenance.event("round", round=1, decision="accepted", valid=0.5)
    provenance.event("summary", initial=0.6, best=0.5)
    events_file = next(tmp_path.rglob("events.jsonl"))
    events = [json.loads(l) for l in open(events_file)]
    assert [e["event"] for e in events] == ["round", "summary"]
    assert events[0]["decision"] == "accepted"


def test_event_log_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ALIS_DWQ_RUN_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("ALIS_DWQ_RUN_LOG", "0")
    monkeypatch.setattr(provenance, "_run_dir", None)
    provenance.event("round", round=1)
    assert list(tmp_path.rglob("events.jsonl")) == []
