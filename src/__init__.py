__all__ = ["make_dataset"]


def __getattr__(name):
	if name == "make_dataset":
		from .make_dataset import make_dataset
		return make_dataset
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
