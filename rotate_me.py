import streamlit as st
from PIL import Image
import os
from transformers import BertTokenizer
from doctr.models import ocr_predictor

from pointless_deskew_text_analyzer import PointlessDeskewTextAnalyzer
from pointless_deskew_visualizations import PointlessDeskewImageVisualizer
from pointless_deskew_image_processor import PointlessDeskewImageProcessor


def app():

    #######################
    # Load Bert Tokenizer #
    #######################
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    #############################################################
    # How much will the image potentially need to be corrected? #
    #############################################################
    allow_up_to_180_degrees = st.radio(
        "Is it necessary to correct the orientation of the image?", (False, True)
    )

    ########################
    # Select the OCR model #
    ########################
    model_choice = st.radio(
        "Main source of text (machine vs. handwritten):",
        ("Printed texts (fast)", "Able to deal with hand-written texts (slow)"),
    )

    if model_choice == "Printed texts (fast)":
        predictor = ocr_predictor(
            "db_mobilenet_v3_large", "crnn_mobilenet_v3_small", pretrained=True
        )
    else:  # db_resnet50 + parseq
        predictor = ocr_predictor("db_resnet50", "parseq", pretrained=True)

    st.write("Select one of these images")
    folder_path = "test_images"  # Change this to your images folder path

    ##################################################################
    # Display the image grid of images within the test_images folder #
    ##################################################################
    image_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith((".JPG", ".jpg", ".jpeg", ".png"))
    ]
    images = [Image.open(os.path.join(folder_path, f)) for f in image_files]

    if "selected_image_path" not in st.session_state:
        st.session_state["selected_image_path"] = None

    num_columns = 7
    num_rows = len(images) // num_columns + (1 if len(images) % num_columns > 0 else 0)

    for row in range(num_rows):
        cols = st.columns(num_columns)
        for col, idx in zip(cols, range(row * num_columns, (row + 1) * num_columns)):
            if idx < len(images):
                with col:
                    st.image(images[idx], width=100, caption=image_files[idx])
                    if st.button("Select", key=image_files[idx]):
                        st.session_state["selected_image_path"] = os.path.join(
                            folder_path, image_files[idx]
                        )
                        st.write(f"Selected: {image_files[idx]}")

    uploaded_image = st.file_uploader(
        "...or upload your image", type=["JPG", "jpg", "jpeg", "png"]
    )

    ###########################
    # Upload an image to test #
    ###########################
    with st.form("upload_form"):
        image_to_process = None
        if (
            "selected_image_path" in st.session_state
            and st.session_state["selected_image_path"]
        ):
            selected_image = Image.open(st.session_state["selected_image_path"])
            st.image(selected_image, caption="Selected Image", use_column_width=True)
            image_to_process = st.session_state["selected_image_path"]
        elif uploaded_image is not None:
            selected_image = Image.open(uploaded_image)
            st.image(selected_image, caption="Uploaded Image", use_column_width=True)
            image_to_process = uploaded_image

        upload_button = st.form_submit_button("Process Image")

    ####################
    # Deskew the image #
    ####################
    if upload_button and image_to_process:
        # Create instances of text_analyzer, visualizer and img_processor
        text_analyzer = PointlessDeskewTextAnalyzer(predictor, tokenizer)
        visualizer = PointlessDeskewImageVisualizer(in_streamlit=True)
        img_processor = PointlessDeskewImageProcessor(text_analyzer, visualizer)
        # Deskew the uploaded image
        img_processor.process_and_display_image_in_streamlit_app(
            image_to_process, allow_up_to_180_degrees
        )


if __name__ == "__main__":
    app()
