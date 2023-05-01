import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
import pyzipper
from sklearn.preprocessing import PowerTransformer

import warnings
warnings.filterwarnings("ignore")

def decrypt_data():
    # Decrypt the ZIP file with the password
    with open('data/data.zip', 'rb') as f_in:
        with pyzipper.AESZipFile(f_in) as f_zip:
            f_zip.setpassword(bytes(st.secrets.my_cool_secrets['pwd'],'UTF-8'))
            for name in f_zip.namelist():
                data = f_zip.read(name)
                with open(name, 'wb') as f_out:
                    f_out.write(data)
    return 1

@st.cache_data
def load_data():
    decrypt_data()
    df = pd.read_parquet("data.parquet")
    emb=np.array(df['embedding'].to_list())
    emb=PowerTransformer().fit_transform(emb)

    return df,emb

def set_page_config():
    st.set_page_config(
        page_title="شخصیات بندی شدہ غزل کی تجویز کنندہ",
        # layout="centered",
        initial_sidebar_state="collapsed",
        layout="wide",
        menu_items={
        'About': """
                Every time you open the app it will start by giving
                recommendation’s randomly but if you give it feedback at the bottom
                the next recommendation will be made using that information.
                And it will retain this information until you hit refresh or close the
                tab and open it again
        """
    }
    )


def set_custom_css():
    st.markdown(
        """
        <style>
       @font-face {
                font-family: 'Noto Nastaleeq Urdu';
                src: 'fonts/NotoNastaliqUrdu-VariableFont_wght.ttf';
                }

        body {
            font-family: 'Noto Nastaleeq Urdu';
            background-color: #000000;
            color: #FFFFFF;
        }
        .urdu_text{
                    font-family: 'Noto Nastaleeq Urdu';
                    text-align: right;
                    line-height: 2.5;
                  }
        
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_buttons(buttons_placeholder,sep):
    sep.write("---")
    # sep.markdown(f'<h5 class="urdu_text">آپ کے اگلے غزل کی تجویز حاصل کرنے کے لئے نیچے دیئے گئے اختیارات میں سے ایک کا انتخاب کریں۔</h5>',
    #                              unsafe_allow_html=True)
    options = {
        "Thumbs Up": "👍",
        "Skip": "⏩",
        "Thumbs Down": "👎",
    }
    selected_option = None
    _,_,_,col1, col2, col3 = buttons_placeholder.columns(6)
    bt1=col1.button(options["Thumbs Up"]+'پسند',)
    bt2=col2.button(options["Skip"]+'چھوڑیں',)
    bt3=col3.button(options["Thumbs Down"]+'ناپسند',)

    if bt1:
        selected_option = "Thumbs Up"
    elif bt2:
        selected_option = "Skip"
    elif bt3:
        selected_option = "Thumbs Down"
    return selected_option


def update_choices(selected_option):
    st.session_state.choices[selected_option].append(st.session_state.idx)


def display_recommendations(title_placeholder,author_placeholder,
                   sep_1_placeholder,text_placeholder):
    title_placeholder.markdown(f'<h1 class="urdu_text">{st.session_state.title}</h1>',
                                 unsafe_allow_html=True)
    author_placeholder.markdown(f'<h5 class="urdu_text">{st.session_state.author}</h5>',
                                 unsafe_allow_html=True)
    sep_1_placeholder.write('---')
    text_placeholder.markdown(f'<h5 class="urdu_text">{st.session_state.ghazal}</h5>',
                                 unsafe_allow_html=True)
    

def init_choice_tracker():
    if 'choices' not in st.session_state:
        st.session_state.choices = {
            "Thumbs Up": [],
            "Skip": [],
            "Thumbs Down": []
        }

def filter_choices(choices):
    if len(st.session_state.choices["Thumbs Down"])>0:
        for i in st.session_state.choices["Thumbs Down"]:
            choices.remove(i)
    if len(st.session_state.choices["Skip"])>0:
        for i in st.session_state.choices["Skip"]:
            choices.remove(i)
    if len(st.session_state.choices["Thumbs Up"])>0:
        for i in st.session_state.choices["Thumbs Up"]:
            choices.remove(i)
    return choices

def get_recommendation(emb):
    # print(st.session_state.choices)
    y = np.zeros(len(emb))
    for i in st.session_state.choices["Thumbs Up"]:
        y[i] = 1
    clf = svm.LinearSVC(class_weight='balanced',
                        verbose=False,
                        max_iter=10_000, tol=1e-6, C=0.1,random_state=8)
    clf.fit(emb, y) # train
    # infer on whatever data you wish, e.g. the original data
    similarities = clf.decision_function(emb)
    sorted_ix = np.argsort(-similarities).tolist()
    sorted_ix=filter_choices(sorted_ix)
    normalized_distance = np.abs(similarities[sorted_ix[0]]) / np.max(np.abs(similarities))
    certainty_score = int((1 - normalized_distance) * 100)
    # print(certainty_score,sorted_ix[0])
    # proba=clf.predict_proba(emb[sorted_ix[1]].reshape(1, -1))
    return certainty_score,sorted_ix[0],

def get_ghazal(df,emb,title_placeholder,author_placeholder,
                   sep_1_placeholder,text_placeholder):
    # print(st.session_state.choices["Thumbs Up"])
    #we are going to cold start the user with a random ghazal
    if len(st.session_state.choices["Thumbs Up"])<1:
        choices=filter_choices(df.index.values.tolist())
        st.session_state.idx=np.random.choice(choices,size=1)[0]
        st.session_state.proba=0
    else:
        # print('yay ml engaged!!!')
        st.session_state.proba,st.session_state.idx=get_recommendation(emb)
    st.session_state.title=df.iloc[st.session_state.idx].title.strip()
    st.session_state.author=df.iloc[st.session_state.idx].author.strip()
    st.session_state.ghazal=df.iloc[st.session_state.idx].text.replace('\n','<br>').strip()
    #display the ghazal
    display_recommendations(title_placeholder,author_placeholder,
                   sep_1_placeholder,text_placeholder)
    return 1
def setup_carcass():
    title_placeholder = st.empty()
    col_1,col_2=st.columns(2)
    ml_score=col_1
    author_placeholder = col_2
    sep_1_placeholder = st.empty()
    text_placeholder = st.empty()
    sep_2_placeholder = st.empty()
    buttons_placeholder = st.empty()

    return [title_placeholder,author_placeholder,sep_1_placeholder,
            text_placeholder,sep_2_placeholder,buttons_placeholder,ml_score]
def main():

    #setup app
    set_page_config()
    set_custom_css()
    init_choice_tracker()
    #load data
    df,emb=load_data()

    (title_placeholder,author_placeholder,sep_1_placeholder,
    text_placeholder,sep_2_placeholder,buttons_placeholder,ml_score)=setup_carcass()
    selected_option = display_buttons(buttons_placeholder,sep_2_placeholder)
    #get ghazal
    if selected_option:
        update_choices(selected_option)
        get_ghazal(df,emb,title_placeholder,author_placeholder,
                    sep_1_placeholder,text_placeholder)
        
    else:
        get_ghazal(df,emb,title_placeholder,author_placeholder,
                   sep_1_placeholder,text_placeholder)
    ml_score.progress(st.session_state.proba,"مصنوعی ذہانت کی یہ تجویز کی  یقینیت")
    # display_user_choices()

        



if __name__ == "__main__":
    main()
