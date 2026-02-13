def sample_max(samples, query):
    # TODO
    raise NotImplementedError("sample_max is not implemented yet")


def sample_mean_topk(samples, query, k):
    # TODO
    raise NotImplementedError("sample_mean_topk is not implemented yet")


THRESHOLD_METHODS = {
    "sample_max": sample_max,
    "sample_mean_topk": sample_mean_topk,
}
