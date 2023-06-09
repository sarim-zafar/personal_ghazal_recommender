import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import PowerTransformer
import pyzipper

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
    emb=emb.astype(np.float32)

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
       @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap'); 
       @font-face {
                font-family: 'Noto Nastaleeq Urdu';
                src: 'fonts/NotoNastaliqUrdu-VariableFont_wght.ttf';
                }

        
        .urdu_text{
                    font-family: 'Noto Nastaleeq Urdu';
                    text-align: right;
                    line-height: 2.5;
                  }
        .eng_text{
                    font-family: 'Roboto', sans-serif; 
                    text-align: left;
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
    txt_col1,txt_col2=text_placeholder.columns(2)
    txt_col1.markdown(f'<h5 class="eng_text">{st.session_state.ghazal_eng}</h5>',
                                 unsafe_allow_html=True)
    txt_col2.markdown(f'<h5 class="urdu_text">{st.session_state.ghazal}</h5>',
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

# Create an Exemplar SVM training function
@st.cache_data
def train_one(X,i,C):
    y = np.zeros(len(X))
    y[i] = 1
    clf = svm.LinearSVC(class_weight='balanced',
                    verbose=False,
                    max_iter=10_000, tol=1e-6, C=C,random_state=8)
    clf.fit(X, y)
    return clf

# @st.cache_data
def train_exemplar_svm(X,pos_examples, C=0.25):
    exemplar_svms = []
    for i in pos_examples:
        # print("training exemplar svm for ",i)
        clf=train_one(X,i,C)
        exemplar_svms.append(clf)
    # print("trained ",len(exemplar_svms),"exemplar svms")
    return exemplar_svms

def normalize_decision_values(decision_values):
    decision_values -= np.max(decision_values)
    return np.exp(decision_values) / np.sum(np.exp(decision_values))

def get_most_conf(top_preds):
    max_idx, max_score = max(top_preds, key=lambda x: x[1])
    return [max_idx,max_score*100]

def get_prediction(_exemplar_svms,X):
    top_preds=[]
    for svm in _exemplar_svms:
        certainty_scores = normalize_decision_values(svm.decision_function(X))
        sorted_ix = np.argsort(certainty_scores)[::-1].tolist()

        # Filter out already chosen items
        sorted_ix = filter_choices(sorted_ix)

        # Get the best match (highest confidence) for the current SVM
        best_match = sorted_ix[0]
        best_match_confidence = certainty_scores[best_match]

        # Add the best match and its confidence to the top_preds list
        top_preds.append([best_match, best_match_confidence])

    # print(top_preds)
    top_pred,conf=get_most_conf(top_preds)
    # print(top_pred,conf)
    return top_pred, conf

def get_recommendation(emb):
    clfs=train_exemplar_svm(emb,st.session_state.choices["Thumbs Up"]) # train
    # infer on whatever data you wish, e.g. the original data
    sorted_ix,certainty_score=get_prediction(clfs,emb)
    # print(sorted_ix,certainty_score)
    return certainty_score,sorted_ix,

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
    st.session_state.ghazal_eng=df.iloc[st.session_state.idx].eng_text.replace('\n','<br>').strip()
    
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
