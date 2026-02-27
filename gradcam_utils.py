"""
Grad-CAM Utility Module
Generates Gradient-weighted Class Activation Maps for EfficientNetB4.
Produces heatmap overlays with red/yellow/green color regions.
Compatible with Keras 3.x / TensorFlow 2.18+.
"""

import numpy as np
import cv2
import tensorflow as tf
import keras


def get_gradcam_heatmap(model, img_array, last_conv_layer_name=None, pred_index=None):
    """
    Compute Grad-CAM heatmap for a given image.
    Handles nested models (e.g. EfficientNet wrapped inside a top-level model).
    """
    # --- Find the nested base model and its last conv layer ---
    base_model_layer, conv_layer_name = _find_conv_layer_info(model)

    if last_conv_layer_name is not None:
        conv_layer_name = last_conv_layer_name

    # Build an "extended" version of the base model that also outputs
    # the last conv layer activations alongside its normal output.
    conv_layer = base_model_layer.get_layer(conv_layer_name)
    extended_base = keras.Model(
        inputs=base_model_layer.input,
        outputs=[conv_layer.output, base_model_layer.output],
    )

    # Manual forward pass through the top-level model's layers,
    # swapping in the extended base to capture conv activations.
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        x = img_tensor
        conv_out = None

        for layer in model.layers:
            # Skip the InputLayer
            if isinstance(layer, keras.layers.InputLayer):
                continue

            # Swap in extended base to capture conv output
            if layer is base_model_layer:
                conv_out, x = extended_base(x, training=False)
            else:
                # Call layer (pass training=False for BN/Dropout)
                try:
                    x = layer(x, training=False)
                except TypeError:
                    x = layer(x)

        predictions = x
        if pred_index is None:
            pred_index = 0  # binary: single output neuron
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_out)  # (1, h, w, filters)

    if grads is None:
        h, w = int(conv_out.shape[1]), int(conv_out.shape[2])
        return np.ones((h, w), dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

    conv_outputs = conv_out[0]                              # (h, w, filters)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap).numpy()

    # ReLU + normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_gradcam(original_img, heatmap, alpha=0.45, colormap=None):
    """
    Overlay a Grad-CAM heatmap on the original image with
    red (high) → yellow (medium) → green (low) color scale.

    Parameters
    ----------
    original_img : np.ndarray
        Original BGR image (uint8).
    heatmap : np.ndarray
        Grad-CAM heatmap in [0, 1].
    alpha : float
        Overlay transparency.
    colormap : int or None
        OpenCV colormap constant. If None, a custom R/Y/G map is used.

    Returns
    -------
    overlay : np.ndarray
        BGR image (uint8) with heatmap overlay.
    """
    h, w = original_img.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    if colormap is not None:
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    else:
        # Custom Red-Yellow-Green colormap
        colored_heatmap = _apply_ryg_colormap(heatmap_uint8)

    overlay = cv2.addWeighted(original_img, 1 - alpha, colored_heatmap, alpha, 0)
    return overlay, colored_heatmap


def _apply_ryg_colormap(heatmap_uint8):
    """
    Custom Red → Yellow → Green colormap (BGR).
      - 0     → Green  (0, 200, 0)
      - 128   → Yellow (0, 255, 255)
      - 255   → Red    (0, 0, 255)
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        ratio = i / 255.0
        if ratio < 0.5:
            # Green → Yellow  (increase Red channel)
            t = ratio / 0.5
            b, g, r = 0, int(200 + 55 * t), int(255 * t)
        else:
            # Yellow → Red  (decrease Green channel)
            t = (ratio - 0.5) / 0.5
            b, g, r = 0, int(255 * (1 - t)), 255
        lut[i, 0] = [b, g, r]

    # Apply look-up table
    colored = cv2.LUT(cv2.merge([heatmap_uint8, heatmap_uint8, heatmap_uint8]), lut)
    return colored


def _find_conv_layer_info(model):
    """
    Walk backwards through the model to find the last Conv2D layer.
    Returns (base_model, layer_name) — base_model is the sub-model that
    actually contains the conv layer (handles nested Functional models).
    """
    for layer in reversed(model.layers):
        # Handle nested models (e.g., the EfficientNet base wrapped as a layer)
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            for sub_layer in reversed(layer.layers):
                if "conv" in sub_layer.__class__.__name__.lower():
                    return layer, sub_layer.name
        if "conv" in layer.__class__.__name__.lower():
            return model, layer.name
    raise ValueError("Could not find a Conv2D layer in the model.")


def preprocess_image(image_bgr, img_size=380):
    """
    Resize and normalise a BGR image for EfficientNetB4 inference.

    Returns
    -------
    img_array : np.ndarray   – shape (1, img_size, img_size, 3), float32 [0,1]
    img_rgb   : np.ndarray   – resized RGB uint8 for display
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_resized
