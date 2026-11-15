"""The Kaggle API client, authenticated (one place for every tool that pushes datasets or kernels)."""


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    k = KaggleApi()
    k.authenticate()
    return k
