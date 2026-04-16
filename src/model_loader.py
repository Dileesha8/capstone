"""
model_loader.py
---------------
Helper to load legacy Keras (.h5) models that were saved with older Keras
versions (< 2.12).  Two known incompatibilities are handled:

1. Layer names may contain '/' (e.g. 'conv1/conv') – Keras 3 / tf_keras 2.16+
   rejects those.
2. InputLayer config uses 'batch_shape' instead of the newer 'batch_input_shape'
   / 'shape' keyword.

Both are patched transparently via a custom InputLayer subclass that is passed
as a custom_object when loading.
"""
import tensorflow.keras as keras


class _LegacyInputLayer(keras.layers.InputLayer):
    """InputLayer that silently accepts the old 'batch_shape' kwarg."""

    def __init__(self, *args, **kwargs):
        # Rename 'batch_shape' → 'batch_input_shape' (tf_keras ≤2.15 name)
        # or just pop it and pass the shape through, depending on version.
        batch_shape = kwargs.pop("batch_shape", None)
        if batch_shape is not None and "input_shape" not in kwargs and "shape" not in kwargs:
            # Strip the leading batch dimension (None) to get the tensor shape
            kwargs["input_shape"] = tuple(batch_shape[1:])
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Called by model deserialization; same renaming needed here.
        batch_shape = config.pop("batch_shape", None)
        if batch_shape is not None and "input_shape" not in config and "shape" not in config:
            config["input_shape"] = tuple(batch_shape[1:])
        return cls(**config)


def load_legacy_h5(path: str, **kwargs):
    """
    Load a legacy .h5 Keras model, handling old-style configs.

    Parameters
    ----------
    path : str
        Absolute or relative path to the .h5 file.
    **kwargs :
        Extra keyword arguments forwarded to tf_keras.models.load_model,
        e.g. compile=False.

    Returns
    -------
    tf_keras.Model
    """
    custom_objects = kwargs.pop("custom_objects", {})
    custom_objects.setdefault("InputLayer", _LegacyInputLayer)

    return keras.models.load_model(
        path,
        custom_objects=custom_objects,
        **kwargs,
    )
