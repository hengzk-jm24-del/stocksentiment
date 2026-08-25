import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# --- 页面基础配置 ---
st.set_page_config(page_title="Stock Sentiment Dashboard", page_icon="📈", layout="wide")
st.title("📈 Stock Sentiment Analysis Dashboard")
st.write("Compare different machine learning models for stock sentiment classification.")

# --- 侧边栏：选择模型 ---
with st.sidebar:
    st.header("⚙️ Model Selection")
    selected_model = st.radio(
        "Choose a model to train:",
        ("Linear SVC (My Baseline)", "ANN (Teammate's Model)", "Bi-LSTM (Deep Learning)")
    )
    st.info("Label mapping: negative = 0, positive = 1, neutral = 2")

# --- 缓存数据加载 ---
@st.cache_data
def load_data():
    # 读取你们清理好的统一数据集
    df = pd.read_csv('final_cleaned_data.csv').dropna()
    return df

df = load_data()

st.subheader("1. Dataset Overview")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df.head(10), use_container_width=True)
with col2:
    st.write("Class Distribution")
    st.bar_chart(df['Sentiment'].value_counts())
    st.metric("Total Clean Records", f"{len(df):,}")

# --- 训练按钮与核心逻辑 ---
st.subheader(f"2. Training & Evaluating: {selected_model}")

if st.button(f"Train {selected_model}", type="primary"):
    with st.spinner("Training in progress... Please wait."):
        
        X = df['Sentence']
        y = df['Sentiment']
        
        # 统一划分 80% 训练集, 20% 测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        target_names = ['negative', 'neutral', 'positive']
        
        # ==========================================
        # 模型 1: Linear SVC
        # ==========================================
        if selected_model == "Linear SVC (My Baseline)":
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=3, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            
            model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
            model.fit(X_train_vec, y_train)
            preds = model.predict(X_test_vec)
            
            y_test_labels = y_test
            pred_labels = preds

        # ==========================================
        # 模型 2: 朋友的 ANN 模型
        # ==========================================
        elif selected_model == "ANN (Teammate's Model)":
            # 执行朋友的训练集平衡处理 (Undersampling)
            train_df = pd.DataFrame({'Sentence': X_train, 'Sentiment': y_train})
            min_class_size = train_df['Sentiment'].value_counts().min()
            
            train_pos = train_df[train_df['Sentiment'] == 'positive'].sample(n=min_class_size, random_state=42)
            train_neu = train_df[train_df['Sentiment'] == 'neutral'].sample(n=min_class_size, random_state=42)
            train_neg = train_df[train_df['Sentiment'] == 'negative'].sample(n=min_class_size, random_state=42)
            
            balanced_train_df = pd.concat([train_pos, train_neu, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            # 提取 TF-IDF 特征
            vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
            X_train_vec = vectorizer.fit_transform(balanced_train_df['Sentence'])
            X_test_vec = vectorizer.transform(X_test)
            
            # 训练 ANN 模型
            model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', alpha=0.01, solver='adam', max_iter=300, random_state=42, early_stopping=True)
            model.fit(X_train_vec, balanced_train_df['Sentiment'])
            preds = model.predict(X_test_vec)
            
            y_test_labels = y_test
            pred_labels = preds

        # ==========================================
        # 模型 3: Bi-LSTM
        # ==========================================
        elif selected_model == "Bi-LSTM (Deep Learning)":
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train)
            y_test_enc = encoder.transform(y_test)
            target_names = encoder.classes_
            
            max_words = 10000
            max_len = 100
            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(X_train)
            
            X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
            X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)
            
            model = Sequential([
                Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(3, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train_seq, y_train_enc, validation_data=(X_test_seq, y_test_enc), epochs=2, batch_size=32, verbose=0)
            
            pred_probs = model.predict(X_test_seq)
            pred_labels = encoder.inverse_transform(np.argmax(pred_probs, axis=1))
            y_test_labels = encoder.inverse_transform(y_test_enc)

        # --- 结果展示区 ---
        acc = accuracy_score(y_test_labels, pred_labels)
        st.success(f"Training Completed! Accuracy: **{acc:.4f}**")
        
        tab1, tab2 = st.tabs(["📊 Classification Report", "🔥 Confusion Matrix"])
        
        with tab1:
            report_dict = classification_report(y_test_labels, pred_labels, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.round(4), use_container_width=True)
            
        with tab2:
            cm = confusion_matrix(y_test_labels, pred_labels)
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # 使用 ConfusionMatrixDisplay 绘制热力图
            matrix_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
            cmap = 'Blues' if "SVC" in selected_model else ('Greens' if "ANN" in selected_model else 'Oranges')
            matrix_display.plot(ax=ax, cmap=cmap, values_format="d", colorbar=False)
            ax.set_title(f"{selected_model} Confusion Matrix")
            st.pyplot(fig)