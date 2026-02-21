import os
import json
import pandas as pd

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def process_speechocean_data(dataset_path="./dataset"):
    print("🔍 جاري البحث عن ملف scores.json...")
    
    scores_file = find_file("scores.json", dataset_path)
    
    if not scores_file:
        print("❌ ملف scores.json غير موجود في أي مكان داخل مجلد dataset.")
        print("تأكد أن مجلد dataset ليس فارغاً (اكتب ls -la dataset في الـ Terminal للتثبت).")
        return

    print(f"✅ تم العثور على الملف في: {scores_file}")
    
    base_dir = os.path.dirname(scores_file)

    with open(scores_file, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)

    processed_data = []

    print("⚙️ جاري معالجة البيانات وربط الصوت بالتقييم...")
    for utt_id, data in scores_data.items():
        speaker_id = utt_id.split('-')[0]
        wav_path = os.path.join(base_dir, "WAVE", f"SPEAKER{speaker_id}", f"{utt_id}.wav")
        
        target_text = data.get('text', '')
        pronunciation_score = data.get('accuracy', 0)

        processed_data.append({
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "target_text": target_text,
            "score": pronunciation_score,
            "wav_path": wav_path
        })

    df = pd.DataFrame(processed_data)
    output_csv = os.path.join(dataset_path, "clean_metadata.csv")
    df.to_csv(output_csv, index=False)
    
    print(f"✅ تمت معالجة {len(df)} تسجيل صوتي بنجاح!")
    print(f"📁 تم حفظ الجدول النظيف في: {output_csv}")
    print("\nعينة من البيانات:")
    print(df.head(3))

if __name__ == "__main__":
    process_speechocean_data()