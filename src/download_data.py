import os
import urllib.request
import tarfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_and_extract(urls, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    filename = "speechocean762.tar.gz"
    filepath = os.path.join(dest_folder, filename)
    
    if not os.path.exists(filepath):
        downloaded = False
        for url in urls:
            print(f"⏳ جاري محاولة التحميل من السيرفر: {url}")
            try:
                urllib.request.urlretrieve(url, filepath)
                print("✅ تم التحميل بنجاح!")
                downloaded = True
                break
            except Exception as e:
                print(f"❌ السيرفر لا يستجيب ({e})، نمر للسيرفر البديل...")
        
        if not downloaded:
            print("🚨 فشل التحميل من كل الروابط. يرجى التحقق من اتصال الإنترنت.")
            return
    else:
        print(f"✅ الملف {filename} تم تحميله مسبقاً.")
        
    print(f"📦 جاري فك الضغط عن {filename}... (الرجاء الانتظار)")
    try:
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=dest_folder)
        print("🚀 عملية فك الضغط اكتملت! الـ Dataset جاهزة للاستعمال.")
    except Exception as e:
        print(f"❌ حدث خطأ أثناء فك الضغط: {e}")

if __name__ == "__main__":
    DATASET_URLS = [
        "https://us.openslr.org/resources/101/speechocean762.tar.gz",
        "https://openslr.magicdatatech.com/resources/101/speechocean762.tar.gz",
        "http://www.openslr.org/resources/101/speechocean762.tar.gz"
    ]
    DESTINATION = "./dataset"
    
    download_and_extract(DATASET_URLS, DESTINATION)