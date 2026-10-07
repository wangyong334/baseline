"""Per-event result files: one npz per recording, events in the recording's file order.

prob        float32 target probability of every event
decision    optional uint8 final decision (used instead of prob >= threshold when present)
publish_us  optional int64 time at which the decision was published (-1 = unknown)
fingerprint (n_events, sum t, sum x, sum y) of the stream the result belongs to
"""
import json
import os

import numpy as np


def file_name(name):
    return name.replace("\\", "/").replace("/", "__") + ".npz"


def fingerprint(stream):
    return np.array([stream.n_events, int(stream.t.sum()), int(stream.x.sum()), int(stream.y.sum())], dtype=np.int64)


def save_result(directory, stream, prob, publish_us=None, decision=None, meta=None):
    prob = np.asarray(prob, dtype=np.float32)
    if prob.shape != (stream.n_events,):
        raise ValueError("%s: prob has shape %s, expected (%d,)" % (stream.name, prob.shape, stream.n_events))
    fields = {"name": np.array(stream.name), "prob": prob, "fingerprint": fingerprint(stream),
              "meta": np.array(json.dumps(meta or {}, sort_keys=True))}
    if publish_us is not None:
        fields["publish_us"] = np.asarray(publish_us, dtype=np.int64)
        if fields["publish_us"].shape != prob.shape:
            raise ValueError("%s: publish_us shape mismatch" % stream.name)
    if decision is not None:
        fields["decision"] = np.asarray(decision, dtype=np.uint8)
        if fields["decision"].shape != prob.shape:
            raise ValueError("%s: decision shape mismatch" % stream.name)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, file_name(stream.name))
    np.savez(path, **fields)
    return path


def load_result(directory, stream):
    path = os.path.join(directory, file_name(stream.name))
    with np.load(path) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if not np.array_equal(result["fingerprint"], fingerprint(stream)):
        raise ValueError("%s: result does not belong to this stream (fingerprint mismatch)" % stream.name)
    result["meta"] = json.loads(str(result["meta"]))
    return result
