from inference import visualize_predictions,predict,color_mask
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_predictions(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    masks, boxes, pred_cls = predict(img_path, confidence) 

    pil_image = Image.open(img_path)
     
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    for i in range(len(masks)):
        rgb_mask = color_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.6, 0)
        boxes[i][0] = [int(i) for i in boxes[i][0]]
        boxes[i][1] = [int(i) for i in boxes[i][1]]

        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                    thickness=text_th)

    return img

def main():
    st.title("Teeth Segmentation App")

    # Streamlit sidebar for image upload
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        confidence = st.slider('Confidence Threshold', 0.0, 1.0, 0.5)
        rect_th = st.slider('Rectangle Thickness', 1, 10, 2)
        text_size = st.slider('Text Size', 1, 10, 2)
        text_th = st.slider('Text Thickness', 1, 10, 2)

        # Perform predictions and visualization
        result_image = visualize_predictions(uploaded_image, confidence, rect_th, text_size, text_th)

        # Display the result
        st.image(result_image, caption='Segmented Image', use_column_width=True)

if __name__ == "__main__":
    main()



