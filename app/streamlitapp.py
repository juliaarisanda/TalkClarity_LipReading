# Import semua dependensi yang diperlukan
import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set variabel lingkungan PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION ke 'python'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Setel layout aplikasi Streamlit sebagai 'wide'
st.set_page_config(layout='wide')

# Terapkan latar belakang gradasi menggunakan HTML
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(45deg, #f2c1eb, #94bbe9);
            background-size: cover;
            background-attachment: fixed;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Setup sidebar
with st.sidebar:
   
    st.image(image_path, width=250)
    st.title('TalkClarity')
    st.info('Kenali gerakan bibir menjadi sebuah teks dengan membaca dari gerakan bibir.')

# Judul aplikasi
st.title('TalkClarity App Lip Reading')
# Membuat daftar opsi video
options = os.listdir(os.path.join('..', 'data', 's1_converted'))
selected_video = st.selectbox('Pilih Video', options)

# Membuat dua kolom
col1, col2 = st.columns(2)

if options:
    file_path = os.path.join('..', 'data', 's1_converted', selected_video)
    with col1:
        st.info('Animasi GIF Dibawah ini sebuah model Machine Learning saat membuat prediksi')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=50)
        st.image('animation.gif', width=350)

        st.info('Keterangan dibawah ini sebuah output dari model pembelajaran mesin sebagai token')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

    with col2:
        st.info('Video di bawah ini menampilkan video yang dikonversi dalam format mp4')

        os.system(f'ffmpeg -i {file_path} -vcodec libx264 {file_path} -y')

        # Memutar video dalam aplikasi
        video = open(file_path, 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

        st.info('Hasil dari decode token menjadi kata-kata')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
