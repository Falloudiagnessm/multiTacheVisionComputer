import streamlit as st
from PIL import Image
import base64
from transformers import AutoProcessor, AutoModelForCausalLM
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Cache the model and processor to avoid reloading on each run
@st.cache_resource
def load_model_and_processor():
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

# Load the model and processor only once
model, processor = load_model_and_processor()

# Define function to run Florence2 captioning and object detection
def florence2(task_prompt, images, text_input=None):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=images, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(images.width, images.height))

    return parsed_answer

# Function to plot bounding boxes on the image
def plot_bbox(image, data):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox, label in zip(data['bboxes'], data['labels']):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    st.pyplot(fig)

# Streamlit UI
st.title("Florence2 Image Captioning and Disease Detection")

# Image Upload Section
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Sidebar for Caption Generation
    st.sidebar.header("Captioning and Disease Detection")
    
    # Caption Generation Section in Sidebar
    task_prompt_caption = '<MORE_DETAILED_CAPTION>'
    if st.sidebar.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Generate caption using Florence2
            caption = florence2(task_prompt_caption, image)
            
            st.write(f"Generated Caption: {caption}")
    #st.sidebar.write(f"Generated Caption: {caption}")
    # Object Detection Section in Sidebar
    task_prompt_od = '<OD>'
    if st.sidebar.button("Detect Diseases"):
        with st.spinner("Detecting objects..."):
            # Perform object detection using Florence2
            od_results = florence2(task_prompt_od, image)
            #st.sidebar.write(f"Generated Object Detection Results: {od_results}")
            # Plot bounding boxes on the image
            plot_bbox(image, od_results['<OD>'])
