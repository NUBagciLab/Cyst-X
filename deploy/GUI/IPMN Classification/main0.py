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

# ------------------------------------
# Load models
# ------------------------------------
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
# Visualization function (runs on upload)
# ------------------------------------
def visualize_nifti(file_obj):
    img_tensor = load_image(file_obj.name)  # MONAI → tensor

    img_np = img_tensor.numpy().squeeze()   # (96,96,96)
    mid_slice = img_np[:, :, img_np.shape[2] // 2]

    # Normalize to displayable uint8
    slice_norm = (mid_slice - mid_slice.min()) / (mid_slice.max() - mid_slice.min() + 1e-8)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)

    return slice_uint8


# ------------------------------------
# Classification function (button click)
# ------------------------------------
def classify(file_obj, model_choice):

    model = model_t1 if model_choice == "T1 model" else model_t2

    img = load_image(file_obj.name).to(device)

    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy().squeeze()

    prob_dict = {LABELS[i]: float(probs[i]) for i in range(2)}
    return prob_dict


# ------------------------------------
# Gradio Blocks UI
# ------------------------------------
with gr.Blocks(title="3D MRI Risk Classification") as ui:

    gr.Markdown("## 3D MRI Risk Classification (T1/T2)\nUpload a NIfTI file to visualize it immediately.")

    with gr.Row():
        file_input = gr.File(label="Upload NIfTI (.nii or .nii.gz)")
        model_choice = gr.Dropdown(["T1 model", "T2 model"], value="T1 model", label="Choose Model")

    # Visualization output
    img_display = gr.Image(label="Center Slice (auto-displayed)")

    # Prediction output
    prediction_out = gr.Label(label="Prediction")

    # Predict button
    predict_btn = gr.Button("Run Classification")

    # -------------------------
    # Event bindings
    # -------------------------
    file_input.change(fn=visualize_nifti, inputs=file_input, outputs=img_display)
    predict_btn.click(fn=classify, inputs=[file_input, model_choice], outputs=prediction_out)


ui.launch()
