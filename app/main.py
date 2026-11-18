"""Streamlit interface: streamlit run app/main.py"""
from pathlib import Path
import streamlit as st

from app.config import Settings
from app.dataset import load_product_dataset

st.set_page_config(page_title="Product Description Enhancer", page_icon="✍️", layout="centered")
st.title("Product Description Enhancer")
st.caption("Turn product facts into storefront-ready copy with a locally fine-tuned DistilGPT-2 + LoRA model.")

model_dir = Path(Settings.from_env().output_dir)
trained = all((model_dir / relative).is_file() for relative in (
    "adapter_config.json", "adapter.pt", "tokenizer/tokenizer_config.json"
))
if trained:
    st.success(f"Fine-tuned model found at `{model_dir}`")
else:
    st.info("The LoRA adapter has not been trained yet. Train it with `python -m app.training`, then return here to generate descriptions.")
    st.code("pip install -r requirements.txt\npython -m app.training", language="bash")

try:
    categories = sorted({row["category"] for row in load_product_dataset()})
except Exception as exc:
    categories = ["Electronics", "Fashion", "Home & Kitchen", "Fitness", "Stationery"]
    st.error(f"Could not load the product dataset: {exc}")

with st.form("product-description-form"):
    product_name = st.text_input("Product name", placeholder="e.g. Everyday Canvas Backpack")
    category = st.selectbox("Category", categories)
    raw_specs = st.text_area("Specifications", placeholder="Capacity: 20 litres\nMaterial: Cotton canvas\nStorage: Padded laptop sleeve", height=150,
                             help="One key: value pair per line. Include only verified product facts.")
    tone = st.selectbox("Tone", ["professional", "persuasive", "minimalist", "premium"])
    submitted = st.form_submit_button("Generate description", type="primary", disabled=not trained)

if submitted:
    specs = {}
    for line in raw_specs.splitlines():
        if not line.strip(): continue
        if ":" not in line:
            st.error(f"Please format each specification as `Key: value`. Could not read: {line}")
            st.stop()
        key, value = line.split(":", 1)
        if not key.strip() or not value.strip():
            st.error("Each specification needs both a key and a value.")
            st.stop()
        specs[key.strip()] = value.strip()
    if not product_name.strip(): st.error("Enter a product name.")
    elif not specs: st.error("Add at least one product specification.")
    else:
        try:
            from app.inference import generate_description_with_details, load_model
            with st.spinner("Loading the fine-tuned model and writing your description…"):
                bundle = load_model(model_dir)
                result = generate_description_with_details(product_name.strip(), category, specs, tone, bundle)
                description = result["description"]
            if result["used_fallback"]:
                st.info("The LoRA output did not pass the product-name, specification, and wording checks. Showing a specification-grounded fallback instead.")
                with st.expander("Raw LoRA output (before grounding checks)"):
                    st.text(result["model_output"] or "The model returned no text.")
                    st.caption(result["fallback_reason"])
            else:
                st.success("LoRA output preserved the product name and every supplied specification, and passed the conservative wording check.")
            st.subheader("Generated description")
            st.text_area("Your description (select and copy)", value=description, height=180, label_visibility="collapsed")
            st.download_button("Download description", data=description, file_name="product-description.txt", mime="text/plain")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

with st.expander("About this model"):
    st.write("This project trains custom low-rank A/B matrices on DistilGPT-2 attention projections. Original model weights stay frozen. The model runs on CPU by default when CUDA is unavailable; generation and fine-tuning can take several minutes on CPU.")
    st.write("Descriptions are generated from the supplied facts. Review the output for accuracy before publishing.")
