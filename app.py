import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Load the trained model from Google Drive
MODEL_PATH = "idrid_final_model.keras"
class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(*args, **kwargs)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"Dense": CompatibleDense},
    compile=False
)

CLASS_NAMES = [
    "Grade 0 — No Apparent Retinopathy",
    "Grade 1 — Mild Non-Proliferative DR",
    "Grade 2 — Moderate Non-Proliferative DR",
    "Grade 3 — Severe Non-Proliferative DR",
    "Grade 4 — Proliferative DR"
]

CLINICAL_NOTES = {
    0: "No retinal abnormalities detected. Routine annual screening is typically recommended.",
    1: "Microaneurysms only. Early-stage signs detected; periodic monitoring is advised.",
    2: "More than microaneurysms, but less than severe NPDR. Follow-up with an ophthalmologist is recommended.",
    3: "Extensive intraretinal hemorrhages or venous beading. Prompt specialist evaluation is strongly advised.",
    4: "Signs of neovascularization or vitreous hemorrhage. Immediate ophthalmological consultation is required."
}

# 2. Inference function
def predict_dr(image):
    if image is None:
        return None, "⚠️ **No image uploaded.** Please upload a retinal fundus photograph to begin analysis."

    # Preprocess image for ResNet50
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Run prediction
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    probabilities = {CLASS_NAMES[i]: float(preds[i]) for i in range(5)}

    summary_md = f"""
    ### Diagnostic Summary
    * **Predicted Severity:** {CLASS_NAMES[top_idx]}
    * **Model Confidence:** {confidence:.2f}%
    * **Clinical Context:** {CLINICAL_NOTES[top_idx]}
    """
    return probabilities, summary_md

# 3. Custom Professional UI using Gradio Blocks
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter")
)

with gr.Blocks(theme=theme, title="Diabetic Retinopathy Screening Assistant") as demo:
    # Header
    gr.Markdown(
        """
        # 👁️ Diabetic Retinopathy Severity Screening
        **AI-Assisted Fundus Image Classification** | Powered by ResNet50 & IDRiD Dataset
        
        Upload a high-resolution retinal fundus photograph below to evaluate Diabetic Retinopathy (DR) severity across five standardized clinical grades (0–4).
        """
    )

    gr.HTML("<hr style='margin: 10px 0 20px 0; border-color: #e2e8f0;'>")

    # Main Two-Column Layout
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Retinal Fundus Photograph",
                height=340
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear", variant="secondary")
                analyze_btn = gr.Button(value="Analyze Scan", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(
                num_top_classes=5,
                label="Severity Grade Probability Distribution"
            )
            summary_output = gr.Markdown(
                value="### Diagnostic Summary\n*Upload a retinal scan and click **Analyze Scan** to view the assessment.*"
            )

    # Clinical Terminology & Grade Guide
    with gr.Accordion("Understanding Diabetic Retinopathy Grades (0–4)", open=False):
        gr.Markdown(
            """
            **Diabetic Retinopathy (DR)** is a diabetes-related eye condition caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (the retina). It progresses through two main stages:

            * **Non-Proliferative DR (NPDR — Grades 1 to 3):** The early-to-moderate stage where existing retinal blood vessels weaken, bulge, or leak fluid/blood into the retina, without new abnormal vessels growing yet.
            * **Proliferative DR (PDR — Grade 4):** The advanced stage where severe oxygen deprivation triggers the growth of fragile, abnormal new blood vessels (*neovascularization*) that can bleed into the eye and threaten vision.

            | Grade | Classification | Clinical Meaning |
            | :--- | :--- | :--- |
            | **Grade 0** | **No Apparent Retinopathy** | Healthy retina with no visible diabetic vascular abnormalities. |
            | **Grade 1** | **Mild Non-Proliferative DR** | Earliest stage; characterized by small, balloon-like swellings in tiny blood vessels (*microaneurysms*). |
            | **Grade 2** | **Moderate Non-Proliferative DR** | Blood vessels begin to lose their ability to transport blood, leading to retinal hemorrhages and fluid leakage. |
            | **Grade 3** | **Severe Non-Proliferative DR** | Significant blockage of retinal blood supply, causing widespread hemorrhages and signaling the retina to grow new vessels. |
            | **Grade 4** | **Proliferative DR** | Advanced stage involving abnormal new blood vessel growth, carrying a high risk of severe vision loss or retinal detachment. |
            """
        )

    # Collapsible Technical Details & Medical Disclaimer
    with gr.Accordion("Model Specifications & Medical Disclaimer", open=False):
        gr.Markdown(
            """
            * **Architecture:** ResNet50 (Transfer Learning with custom classification head)
            * **Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD)
            * **Input Resolution:** 224 × 224 px (RGB)
            * **Disclaimer:** *This application is a research and educational prototype. It is not a certified medical device and should not be used as a substitute for professional clinical diagnosis or ophthalmological evaluation.*
            """
        )

    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #64748b; font-size: 0.9em;">
            Developed by <strong>Aryan Srivastava</strong> | Contact: <a href="mailto:aryansrivastava737@gmail.com">aryansrivastava737@gmail.com</a>
        </div>
        """
    )

    # Button Actions
    analyze_btn.click(
        fn=predict_dr,
        inputs=image_input,
        outputs=[label_output, summary_output]
    )
    clear_btn.add([image_input, label_output, summary_output])

# Launch with a public shareable link
demo.launch(share=True)
