import numpy as np
import tensorflow as tf
import cv2
import base64

def get_last_conv_layer(model):
    """Auto-detect the last Conv2D layer before GAP."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # fallback: last Activation or BatchNorm before GAP
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Activation,
                               tf.keras.layers.BatchNormalization)):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generate Grad-CAM heatmap using the specified conv layer."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap


def generate_sidebyside_b64(img_path, heatmap, alpha=0.45):
    """
    Returns base64-encoded JPEG of [Original | Grad-CAM overlay] side by side.
    No file saved to disk.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Resize & colorize heatmap
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

    # Add label bars
    font    = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 32

    orig_panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    orig_panel[label_h:] = img_rgb
    cv2.rectangle(orig_panel, (0, 0), (w, label_h), (30, 30, 30), -1)
    cv2.putText(orig_panel, "Original", (8, 22), font, 0.65, (255, 255, 255), 2)

    heat_panel = np.zeros((h + label_h, w, 3), dtype=np.uint8)
    heat_panel[label_h:] = overlay
    cv2.rectangle(heat_panel, (0, 0), (w, label_h), (30, 30, 30), -1)
    cv2.putText(heat_panel, "Grad-CAM Heatmap", (8, 22), font, 0.65, (255, 255, 255), 2)

    # Combine side by side
    combined     = np.hstack([orig_panel, heat_panel])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"