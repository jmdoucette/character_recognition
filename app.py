#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:40:20 2021

@author: jamesdoucette
"""
import pandas as pd
import numpy as np
import tensorflow as tf

from model import *

from PIL import Image
import streamlit as st
from skimage.util import img_as_float,img_as_ubyte
from skimage import color
from  skimage.transform import resize
from streamlit_drawable_canvas import st_canvas

def convert_image(im):
    return img_as_ubyte(resize(color.rgba2rgb(img_as_ubyte(im)), (28, 28),anti_aliasing=True))


stroke_width = 25
stroke_color = '#fff'
bg_color = "#000"

realtime_update = True

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color= bg_color,
    background_image= None,
    update_streamlit=realtime_update,
    height=600,
    drawing_mode="freedraw",
    key="canvas",
)
predict=st.button('predict')
if predict:
    im=canvas_result.image_data

    model = tf.keras.models.load_model('model')
    converted = convert_image(im)
    st.write(make_prediction(model,converted))





