import gradio as gr
import torch
import numpy as np
from model import get_model
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, Compose, ScaleIntensity

# ------------------------------------
# Device
# ------------------------------------
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

LABELS = ['no/low risk', 'high risk']

load_image = Compose([
    LoadImage(image_only=True),
    ScaleIntensity(),
    EnsureChannelFirst(),
    Resize((96, 96, 96))
])

# ----------------------------
# Load T1/T2 models
# ----------------------------
def load_model(path):
    m = get_model(out_channels=2)
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

print("Loading models...")
model_t1 = load_model("./model_t1.pth")
model_t2 = load_model("./model_t2.pth")
print("Models loaded.")


# ------------------------------------
# Visualization helper
# ------------------------------------
def get_slice(volume, axis, index):
    if volume is None:
        return None

    if axis == "z":      # axial
        sl = volume[:, :, index]
    elif axis == "y":    # coronal
        sl = volume[:, index, :]
    else:                # axis == "x", sagittal
        sl = volume[index, :, :]

    # Normalize to 0-255 uint8
    norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    img = (norm * 255).astype(np.uint8)
    return img


# ------------------------------------
# Load volume on upload
# ------------------------------------
def load_volume(file_obj):
    if file_obj is None:
        return None, None, None, None

    img_tensor = load_image(file_obj.name)  # (1,96,96,96)
    vol = img_tensor.numpy().squeeze()      # (96,96,96)

    # Default slice indices (center for each view)
    mid = vol.shape[0] // 2

    axial   = get_slice(vol, "z", mid)
    coronal = get_slice(vol, "y", mid)
    sagittal = get_slice(vol, "x", mid)

    return vol, axial, coronal, sagittal


# ------------------------------------
# Classification
# ------------------------------------
def classify(file_obj, model_choice):

    model = model_t1 if model_choice == "T1 model" else model_t2
    img = load_image(file_obj.name).to(device)

    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy().squeeze()

    return {
        LABELS[0]: float(probs[0]),
        LABELS[1]: float(probs[1]),
    }


# ------------------------------------
# Gradio UI with 3-plane visualization
# ------------------------------------
with gr.Blocks(title="IPMN Risk Classification") as ui:

    gr.Markdown("## Upload a NIfTI file, select the model, and run classification.")

    with gr.Row():
        file_input = gr.File(label="Upload NIfTI (.nii or .nii.gz)")
        model_choice = gr.Dropdown(["T1 model", "T2 model"], value="T1 model", label="Model")

    # State to store full 3D volume
    volume_state = gr.State()

    # ----- Z-axis (Axial) -----
    gr.Markdown("### Axial (Z-axis) View")
    axial_slider = gr.Slider(0, 95, value=48, step=1, label="Axial Slice Index")
    axial_img = gr.Image(label="Axial Slice")

    # ----- Y-axis (Coronal) -----
    gr.Markdown("### Coronal (Y-axis) View")
    coronal_slider = gr.Slider(0, 95, value=48, step=1, label="Coronal Slice Index")
    coronal_img = gr.Image(label="Coronal Slice")

    # ----- X-axis (Sagittal) -----
    gr.Markdown("### Sagittal (X-axis) View")
    sagittal_slider = gr.Slider(0, 95, value=48, step=1, label="Sagittal Slice Index")
    sagittal_img = gr.Image(label="Sagittal Slice")

    # Prediction output
    predict_btn = gr.Button("Run Classification")
    prediction_out = gr.Label(label="Prediction")

    # -------------------------
    # Bind events
    # -------------------------
    file_input.change(
        fn=load_volume,
        inputs=file_input,
        outputs=[volume_state, axial_img, coronal_img, sagittal_img]
    )

    axial_slider.change(
        fn=lambda vol, idx: get_slice(vol, "z", idx),
        inputs=[volume_state, axial_slider],
        outputs=axial_img
    )

    coronal_slider.change(
        fn=lambda vol, idx: get_slice(vol, "y", idx),
        inputs=[volume_state, coronal_slider],
        outputs=coronal_img
    )

    sagittal_slider.change(
        fn=lambda vol, idx: get_slice(vol, "x", idx),
        inputs=[volume_state, sagittal_slider],
        outputs=sagittal_img
    )

    predict_btn.click(
        fn=classify,
        inputs=[file_input, model_choice],
        outputs=prediction_out
    )

ui.launch()
