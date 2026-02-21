import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

def extract_mfcc(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        return None

def train_model():
    print("🔍 جاري قراءة البيانات (clean_metadata.csv)...")
    csv_path = "./dataset/clean_metadata.csv"
    
    if not os.path.exists(csv_path):
        print("❌ ملف clean_metadata.csv غير موجود!")
        return

    df = pd.read_csv(csv_path)
    
    # --- الفهرسة الذكية (بونتو في الحروف الكبيرة والصغيرة) ---
    print("🚀 جاري مسح مجلد dataset بالكامل للبحث عن ملفات الصوت...")
    wav_dict = {}
    for root, dirs, files in os.walk(os.path.abspath("./dataset")):
        for file in files:
            # النقطة السحرية: نردوها lower() باش يقبل .wav و .WAV
            if file.lower().endswith(".wav"): 
                wav_dict[file.lower()] = os.path.join(root, file)
                
    print(f"✅ تم العثور على {len(wav_dict)} ملف صوتي إجمالاً في الجهاز.")

    X, y = [], []
    missing_files = 0

    print("🎧 جاري استخراج الخصائص الصوتية... (هذا سيستغرق بعض الوقت)")
    
    for index, row in df.iterrows():
        # ناخذو اسم الملف ونردوه حروف صغيرة باش يتطابق مع الفهرس
        filename = os.path.basename(row['wav_path']).lower()
        
        real_wav_path = wav_dict.get(filename)
        
        if real_wav_path is None:
            missing_files += 1
            continue
            
        features = extract_mfcc(real_wav_path)
        if features is not None:
            X.append(features)
            normalized_score = (row['score'] / 10.0) * 100
            y.append(normalized_score)
            
        if len(X) % 500 == 0 and len(X) > 0:
            print(f"✅ تم تحميل ومعالجة {len(X)} ملف صوتي...")

    print(f"ℹ️ النتيجة: تم معالجة {len(X)} ملف. الملفات المفقودة: {missing_files}")
    
    if len(X) == 0:
        print("🚨 خطأ: لم يتم معالجة أي ملف. تأكد أن الملفات المفكوك ضغطها موجودة داخل مجلد dataset.")
        return

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 جاري بناء وتدريب الـ CNN...")
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(40, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    checkpoint = ModelCheckpoint('dataset/speech_cnn_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(
        X_train, y_train,
        epochs=30, 
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    print("🎉 اكتمل التدريب! تم حفظ الموديل الذكي في: dataset/speech_cnn_model.h5")

if __name__ == "__main__":
    train_model()