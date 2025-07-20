import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import difflib
import pandas as pd
import sys
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings
import string

# إخفاء التحذيرات غير الضرورية
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# تكوين إعدادات النظام للنص العربي والإنجليزي
sys.stdout.reconfigure(encoding='utf-8')

# تحميل القاموس الطبي ثنائي اللغة
MEDICAL_DICTIONARY = []
MEDICAL_SYNONYMS = {}
COMMON_WORDS = set()  # كلمات شائعة غير دوائية

def load_medical_resources():
    global MEDICAL_DICTIONARY, MEDICAL_SYNONYMS, COMMON_WORDS
    
    try:
        # تحميل القاموس الطبي
        if os.path.exists("medical_dictionary.xlsx"):
            df = pd.read_excel("medical_dictionary.xlsx")
            MEDICAL_DICTIONARY = df['Medicine'].dropna().str.strip().str.lower().tolist()
            
            # تحميل المرادفات
            if 'Synonyms' in df.columns:
                for idx, row in df.iterrows():
                    if pd.notna(row['Synonyms']):
                        synonyms = [s.strip().lower() for s in row['Synonyms'].split(',')]
                        for synonym in synonyms:
                            MEDICAL_SYNONYMS[synonym] = row['Medicine'].strip().lower()
        
        # تحميل الكلمات الشائعة غير الدوائية
        if os.path.exists("common_non_drug_words.txt"):
            with open("common_non_drug_words.txt", "r", encoding="utf-8") as f:
                COMMON_WORDS = set(line.strip().lower() for line in f)
        else:
            # قائمة افتراضية بالكلمات الشائعة غير الدوائية
            COMMON_WORDS = {
                "المريض", "التاريخ", "العمر", "الوزن", "التشخيص", "الروشتة", "الطبيب",
                "العيادة", "المستشفى", "الاسم", "الجرعة", "المدة", "الاستخدام", "الكمية",
                "patient", "date", "age", "weight", "diagnosis", "prescription", "doctor",
                "clinic", "hospital", "name", "dose", "duration", "usage", "quantity",
                "التحليل", "الفحص", "النتيجة", "التقرير", "analysis", "examination", "result", "report"
            }
            with open("common_non_drug_words.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(COMMON_WORDS))
    except Exception as e:
        print(f"[!] تحذير: {e}")
        # قائمة الأدوية الافتراضية ثنائية اللغة
        MEDICAL_DICTIONARY = [
            "باراسيتامول", "ايبوبروفين", "أموكسيسيلين", "سيتريزين", "ميتفورمين",
            "سيبروفلوكساسين", "أوميبرازول", "فيتامين د", "أسبرين", "هيدروكورتيزون",
            "قطرة عين", "قرص", "كبسولة", "كريم", "مرهم", "شراب", "حقنة",
            "paracetamol", "ibuprofen", "amoxicillin", "cetirizine", "metformin",
            "ciprofloxacin", "omeprazole", "vitamin d", "aspirin", "hydrocortisone",
            "eye drops", "tablet", "capsule", "cream", "ointment", "syrup", "injection"
        ]
        MEDICAL_SYNONYMS = {
            "بارسيتامول": "باراسيتامول",
            "ايببروفين": "ايبوبروفين",
            "اموكسيسيلين": "أموكسيسيلين",
            "فيتامين د3": "فيتامين د",
            "panadol": "paracetamol",
            "brufen": "ibuprofen",
            "amoxil": "amoxicillin"
        }
        COMMON_WORDS = {
            "patient", "date", "doctor", "prescription", "diagnosis", "clinic",
            "hospital", "name", "dose", "duration", "quantity", "age", "weight"
        }
        print("تم استخدام المصادر الافتراضية")

# تحميل المصادر عند البدء
load_medical_resources()

# معالجة الصورة
def preprocess_image(image):
    try:
        # تحويل الصورة إلى تدرج الرمادي
        gray = image.convert('L')
        
        # زيادة حدة الصورة
        sharpened = gray.filter(ImageFilter.SHARPEN)
        
        # تحسين التباين
        enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = enhancer.enhance(2.0)
        
        # تحسين السطوع
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        final_image = brightness_enhancer.enhance(1.2)
        
        return final_image
    except Exception as e:
        print(f"خطأ في معالجة الصورة: {str(e)}")
        return image

# تصحيح الكلمات الطبية بدعم ثنائي اللغة
def correct_medical_term(term):
    term = term.strip().lower()
    
    # التحقق من المرادفات أولاً
    if term in MEDICAL_SYNONYMS:
        return MEDICAL_SYNONYMS[term]
    
    # البحث المباشر في القاموس
    if term in MEDICAL_DICTIONARY:
        return term
        
    # استخدام difflib للتقريب
    matches = difflib.get_close_matches(term, MEDICAL_DICTIONARY, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    
    return term

# نموذج التعلم الآلي للتعرف على الأدوية مع تحسينات
class AdvancedDrugRecognitionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1500)
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.trained = False
        
    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)
        self.trained = True
        return "تم تدريب النموذج بنجاح"
    
    def predict(self, text):
        if not self.trained:
            return False
        vec = self.vectorizer.transform([text])
        return self.model.predict(vec)[0] == 1
    
    def save(self, filename):
        joblib.dump((self.vectorizer, self.model), filename)
    
    def load(self, filename):
        try:
            self.vectorizer, self.model = joblib.load(filename)
            self.trained = True
            return True
        except:
            return False

# إنشاء وتدريب نموذج التعرف على الأدوية
drug_model = AdvancedDrugRecognitionModel()

# محاولة تحميل نموذج مدرب مسبقاً
if not drug_model.load("advanced_drug_recognition_model.pkl"):
    # إنشاء بيانات تدريبية شاملة ثنائية اللغة
    train_data = [
        # أدوية (إيجابية)
        ("باراسيتامول", 1), ("ايبوبروفين", 1), ("أموكسيسيلين", 1), 
        ("ميتفورمين", 1), ("فيتامين د", 1), ("عقار", 1), ("دواء", 1),
        ("paracetamol", 1), ("ibuprofen", 1), ("amoxicillin", 1),
        ("metformin", 1), ("vitamin d", 1), ("drug", 1), ("medicine", 1),
        
        # كلمات شائعة ليست أدوية (سلبية)
        ("المريض", 0), ("التاريخ", 0), ("الطبيب", 0), ("الجرعة", 0),
        ("patient", 0), ("date", 0), ("doctor", 0), ("dose", 0),
        ("التشخيص", 0), ("الكمية", 0), ("المدة", 0), ("الاستخدام", 0),
        ("diagnosis", 0), ("quantity", 0), ("duration", 0), ("usage", 0),
        ("الاسم", 0), ("العمر", 0), ("الوزن", 0), ("العيادة", 0),
        ("name", 0), ("age", 0), ("weight", 0), ("clinic", 0),
        ("الروشتة", 0), ("التقرير", 0), ("الفحص", 0), ("التحليل", 0),
        ("prescription", 0), ("report", 0), ("exam", 0), ("analysis", 0),
        
        # كلمات إضافية سلبية
        ("يومياً", 0), ("مرتين", 0), ("كبسولة", 0), ("قرص", 0),
        ("daily", 0), ("twice", 0), ("capsule", 0), ("tablet", 0),
        ("قبل", 0), ("بعد", 0), ("مع", 0), ("الطعام", 0),
        ("before", 0), ("after", 0), ("with", 0), ("food", 0)
    ]
    
    # تحويل البيانات إلى صيغة التدريب
    X = [text for text, label in train_data]
    y = [label for text, label in train_data]
    
    # تدريب النموذج
    drug_model.train(X, y)
    drug_model.save("advanced_drug_recognition_model.pkl")
    print("تم تدريب نموذج التعرف المتقدم على الأدوية")

# استخراج الأدوية مع تحسينات الدقة
def extract_drugs_with_high_accuracy(text):
    # تجميع الكلمات المحتملة
    candidate_words = []
    
    # استخراج جميع الكلمات والمركبات
    words = re.findall(r'\b[\w\s-]{4,}\b', text)  # الكلمات التي تحتوي على 4 أحرف/رموز على الأقل
    
    for word in words:
        word = word.strip().lower()
        
        # تجاهل الكلمات الشائعة غير الدوائية
        if word in COMMON_WORDS:
            continue
            
        # تجاهل الكلمات التي تحتوي على أرقام فقط
        if re.fullmatch(r'[\d\s-]+', word):
            continue
            
        # تجاهل الكلمات القصيرة جداً (أقل من 4 أحرف بعد التنظيف)
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) < 4:
            continue
            
        # إذا كانت الكلمة في القاموس، نعتبرها دواء
        if correct_medical_term(word) in MEDICAL_DICTIONARY:
            candidate_words.append(word)
            continue
            
        # استخدام نموذج الذكاء الاصطناعي للتعرف على الأدوية
        if drug_model.predict(word):
            candidate_words.append(word)
    
    # تجميع المركبات الدوائية (كلمات متعددة)
    compounds = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip().lower()
        if not line or len(line) < 10:
            continue
            
        # تجاهل الأسطر التي تحتوي على كلمات شائعة غير دوائية
        if any(common_word in line for common_word in COMMON_WORDS):
            continue
            
        # البحث عن مركبات دوائية
        if drug_model.predict(line):
            compounds.append(line)
    
    # الجمع بين النتائج وإزالة التكرارات
    all_drugs = list(set(candidate_words + compounds))
    
    # ترتيب النتائج حسب الطول (الأطول أولاً)
    all_drugs.sort(key=len, reverse=True)
    
    return all_drugs

# استخراج تعليمات الاستخدام بدقة
def extract_instructions(text):
    # كلمات دالة على التعليمات
    instruction_keywords = {
        "يومياً", "يوميا", "مرة", "مرتين", "ثلاث", "أربع", "خمس", "ست",
        "قبل", "بعد", "مع", "الطعام", "النوم", "الصباح", "المساء",
        "قرص", "كبسولة", "ملعقة", "قطرة", "حقنة", "جرعة", "مقدار",
        "كل", "ساعات", "لمدة", "أيام", "أسبوع", "أسابيع", "شهر", "أشهر",
        "daily", "times", "once", "twice", "three", "four", "before", "after",
        "with", "food", "sleep", "morning", "evening", "tablet", "capsule",
        "spoon", "teaspoon", "drop", "injection", "dose", "every", "hours",
        "for", "days", "weeks", "month", "months"
    }
    
    instructions = set()
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip().lower()
        if not line or len(line) < 10:
            continue
            
        # إذا احتوت السطر على كلمات تعليمات
        if any(keyword in line for keyword in instruction_keywords):
            # تقسيم السطر إلى جمل صغيرة
            parts = re.split(r'[،.:;]', line)
            for part in parts:
                part = part.strip()
                if any(keyword in part for keyword in instruction_keywords) and len(part) > 8:
                    instructions.add(part)
    
    return sorted(instructions)

# تنفيذ OCR بدعم ثنائي اللغة
def run_ocr(image):
    try:
        reader = easyocr.Reader(['ar', 'en'], gpu=False)
        results = reader.readtext(np.array(image), detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as e:
        messagebox.showerror("خطأ في OCR", f"فشل في استخراج النص: {str(e)}")
        return ""

# واجهة المستخدم المتقدمة مع تحسينات
class AdvancedPrescriptionAnalyzer:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.unknown_drugs = []
        
    def setup_ui(self):
        self.root.title("نظام متقدم لتحليل الروشتات الطبية")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f5f5f5")
        
        # إطار العنوان
        title_frame = tk.Frame(self.root, bg="#2c3e50", padx=15, pady=15)
        title_frame.pack(fill=tk.X)
        
        tk.Label(title_frame, text="نظام التعرف الذكي على الأدوية", 
                font=("Arial", 18, "bold"), bg="#2c3e50", fg="white").pack()
        
        tk.Label(title_frame, text="يدعم اللغتين العربية والإنجليزية - دقة عالية في التعرف", 
                font=("Arial", 11), bg="#2c3e50", fg="#ecf0f1").pack(pady=5)
        
        # إطار التحكم
        control_frame = tk.Frame(self.root, bg="#f5f5f5", padx=15, pady=15)
        control_frame.pack(fill=tk.X)
        
        self.btn_load = tk.Button(
            control_frame, 
            text="📷 تحميل صورة الروشتة", 
            command=self.analyze_prescription,
            width=25, 
            height=2, 
            bg="#3498db", 
            fg="white", 
            font=("Arial", 11, "bold"),
            relief=tk.FLAT
        )
        self.btn_load.pack(side=tk.LEFT, padx=10)
        
        self.btn_add_drug = tk.Button(
            control_frame, 
            text="➕ إضافة دواء جديد", 
            command=self.add_new_drug,
            width=15, 
            height=2, 
            bg="#27ae60", 
            fg="white", 
            font=("Arial", 11),
            state=tk.DISABLED
        )
        self.btn_add_drug.pack(side=tk.RIGHT, padx=10)
        
        # إطار النتائج
        result_frame = tk.Frame(self.root, bg="#f5f5f5", padx=15, pady=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # قسم الأدوية المعروفة
        known_frame = tk.LabelFrame(result_frame, text="الأدوية المعروفة", 
                                  font=("Arial", 12, "bold"), bg="#f5f5f5")
        known_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.known_list = tk.Listbox(known_frame, font=("Arial", 11), 
                                    selectmode=tk.SINGLE, height=8)
        scrollbar_known = tk.Scrollbar(known_frame, orient=tk.VERTICAL, command=self.known_list.yview)
        self.known_list.config(yscrollcommand=scrollbar_known.set)
        
        self.known_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_known.pack(side=tk.RIGHT, fill=tk.Y)
        
        # قسم الأدوية غير المعروفة
        unknown_frame = tk.LabelFrame(result_frame, text="الأدوية غير المعروفة (احتمالية)", 
                                    font=("Arial", 12, "bold"), bg="#f5f5f5")
        unknown_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        self.unknown_list = tk.Listbox(unknown_frame, font=("Arial", 11), 
                                      selectmode=tk.SINGLE, height=3)
        scrollbar_unknown = tk.Scrollbar(unknown_frame, orient=tk.VERTICAL, command=self.unknown_list.yview)
        self.unknown_list.config(yscrollcommand=scrollbar_unknown.set)
        
        self.unknown_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_unknown.pack(side=tk.RIGHT, fill=tk.Y)
        
        # قسم التعليمات
        instr_frame = tk.LabelFrame(result_frame, text="تعليمات الاستخدام", 
                                  font=("Arial", 12, "bold"), bg="#f5f5f5")
        instr_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        self.instr_text = tk.Text(instr_frame, height=5, font=("Arial", 11), wrap=tk.WORD)
        scrollbar_instr = tk.Scrollbar(instr_frame, orient=tk.VERTICAL, command=self.instr_text.yview)
        self.instr_text.config(yscrollcommand=scrollbar_instr.set)
        
        self.instr_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_instr.pack(side=tk.RIGHT, fill=tk.Y)
        
        # قسم النص الكامل
        fulltext_frame = tk.LabelFrame(result_frame, text="النص المستخرج", 
                                     font=("Arial", 12, "bold"), bg="#f5f5f5")
        fulltext_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        self.full_text = tk.Text(fulltext_frame, height=10, font=("Arial", 10), wrap=tk.WORD)
        scrollbar_full = tk.Scrollbar(fulltext_frame, orient=tk.VERTICAL, command=self.full_text.yview)
        self.full_text.config(yscrollcommand=scrollbar_full.set)
        
        self.full_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_full.pack(side=tk.RIGHT, fill=tk.Y)
    
    def add_new_drug(self):
        selected_idx = self.unknown_list.curselection()
        if not selected_idx:
            messagebox.showwarning("تحذير", "يرجى اختيار دواء من القائمة")
            return
            
        drug_name = self.unknown_list.get(selected_idx[0])
        
        # نافذة إضافة الدواء
        dialog = tk.Toplevel(self.root)
        dialog.title("إضافة دواء جديد")
        dialog.geometry("500x400")
        dialog.grab_set()
        
        tk.Label(dialog, text="اسم الدواء:", font=("Arial", 11)).pack(pady=5)
        name_entry = tk.Entry(dialog, font=("Arial", 11), width=40)
        name_entry.insert(0, drug_name)
        name_entry.pack(pady=5)
        
        tk.Label(dialog, text="المرادفات (مفصولة بفواصل):", font=("Arial", 11)).pack(pady=5)
        synonyms_entry = tk.Entry(dialog, font=("Arial", 11), width=40)
        synonyms_entry.pack(pady=5)
        
        tk.Label(dialog, text="اللغة:", font=("Arial", 11)).pack(pady=5)
        lang_var = tk.StringVar(value="ar")
        tk.Radiobutton(dialog, text="العربية", variable=lang_var, value="ar", font=("Arial", 11)).pack()
        tk.Radiobutton(dialog, text="الإنجليزية", variable=lang_var, value="en", font=("Arial", 11)).pack()
        
        def save_drug():
            name = name_entry.get().strip()
            synonyms = [s.strip().lower() for s in synonyms_entry.get().split(",") if s.strip()]
            lang = lang_var.get()
            
            if not name:
                messagebox.showwarning("تحذير", "يرجى إدخال اسم الدواء")
                return
                
            # تحديث القاموس
            if name.lower() not in MEDICAL_DICTIONARY:
                MEDICAL_DICTIONARY.append(name.lower())
                
            # تحديث المرادفات
            for synonym in synonyms:
                if synonym and synonym not in MEDICAL_SYNONYMS:
                    MEDICAL_SYNONYMS[synonym] = name.lower()
            
            # تحديث ملف Excel
            self.update_dictionary_file(name, synonyms, lang)
            
            # تحديث الواجهة
            self.unknown_list.delete(selected_idx[0])
            self.known_list.insert(tk.END, name)
            
            messagebox.showinfo("تم", f"تم إضافة {name} إلى القاموس الطبي")
            dialog.destroy()
        
        tk.Button(dialog, text="حفظ", command=save_drug, 
                 font=("Arial", 11), bg="#27ae60", fg="white", width=15).pack(pady=15)
    
    def update_dictionary_file(self, name, synonyms, lang="ar"):
        try:
            if os.path.exists("medical_dictionary.xlsx"):
                df = pd.read_excel("medical_dictionary.xlsx")
            else:
                df = pd.DataFrame(columns=["Medicine", "Synonyms", "Language"])
            
            # تحقق إذا كان الدواء موجوداً بالفعل
            if name.lower() not in df['Medicine'].str.lower().tolist():
                new_row = {
                    "Medicine": name,
                    "Synonyms": ", ".join(synonyms),
                    "Language": lang
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_excel("medical_dictionary.xlsx", index=False)
        except Exception as e:
            print(f"خطأ في تحديث ملف القاموس: {str(e)}")
    
    def analyze_prescription(self):
        file_types = [
            ("ملفات الصور", "*.jpg *.jpeg *.png *.bmp"),
            ("جميع الملفات", "*.*")
        ]
        
        try:
            file_path = filedialog.askopenfilename(filetypes=file_types)
            if not file_path:
                return
                
            # تحديث واجهة المستخدم أثناء المعالجة
            self.btn_load.config(state=tk.DISABLED, text="جارٍ التحليل...")
            self.known_list.delete(0, tk.END)
            self.unknown_list.delete(0, tk.END)
            self.instr_text.delete(1.0, tk.END)
            self.full_text.delete(1.0, tk.END)
            self.full_text.insert(tk.END, "جارٍ معالجة الصورة...")
            self.root.update()
            
            try:
                # تحميل الصورة
                image = Image.open(file_path)
                
                # معالجة الصورة
                processed = preprocess_image(image)
                
                # استخراج النص
                text = run_ocr(processed)
                self.full_text.delete(1.0, tk.END)
                self.full_text.insert(tk.END, text)
                
                if not text.strip():
                    messagebox.showwarning("تحذير", "لم يتم العثور على نص في الصورة")
                    return
                
                # استخراج الأدوية باستخدام الذكاء الاصطناعي
                drugs = extract_drugs_with_high_accuracy(text)
                
                # استخراج التعليمات
                instructions = extract_instructions(text)
                
                # فصل الأدوية المعروفة وغير المعروفة
                known_drugs = []
                unknown_drugs = []
                
                for drug in drugs:
                    if drug in MEDICAL_DICTIONARY or correct_medical_term(drug) in MEDICAL_DICTIONARY:
                        known_drugs.append(drug)
                    else:
                        unknown_drugs.append(drug)
                
                # عرض النتائج
                for drug in known_drugs:
                    self.known_list.insert(tk.END, drug)
                
                for drug in unknown_drugs:
                    self.unknown_list.insert(tk.END, drug)
                
                if instructions:
                    self.instr_text.delete(1.0, tk.END)
                    self.instr_text.insert(tk.END, "\n".join(f"• {instr}" for instr in instructions))
                else:
                    self.instr_text.delete(1.0, tk.END)
                    self.instr_text.insert(tk.END, "لم يتم التعرف على تعليمات واضحة")
                
                # تفعيل زر إضافة الأدوية إذا كانت هناك أدوية غير معروفة
                self.unknown_drugs = unknown_drugs
                self.btn_add_drug.config(state=tk.NORMAL if unknown_drugs else tk.DISABLED)
                
                messagebox.showinfo("اكتمل التحليل", 
                                  f"تم التعرف على {len(known_drugs)} دواء معروف\nو {len(unknown_drugs)} دواء محتمل")
                
            except Exception as e:
                messagebox.showerror("خطأ", f"حدث خطأ أثناء المعالجة:\n{str(e)}")
        finally:
            self.btn_load.config(state=tk.NORMAL, text="📷 تحميل صورة الروشتة")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPrescriptionAnalyzer(root)
    root.mainloop()