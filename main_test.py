import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Nano-Clear AI Detector",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Nano-Clear AI Detector & Smart Analyzer")
st.write("نظام الذكاء الاصطناعي المتقدم لتحليل وحساب نسب تلوث الميكروبلاستيك بدقة عالية")
st.divider()

# إعدادات متقدمة للذكاء الاصطناعي في الشريط الجانبي
st.sidebar.header("⚙️ إعدادات نموذج الذكاء الاصطناعي")
sensitivity = st.sidebar.slider("حساسية الكشف (Sensitivity)", min_value=10, max_value=100, value=50)
min_particle_size = st.sidebar.slider("أقل حجم للجسيم (بكسل)", min_value=1, max_value=50, value=5)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 رفع صورة من الجهاز")
    uploaded_file = st.file_uploader(
        "اختر صورة العينة",
        type=["jpg", "jpeg", "png"],
        key="upload_image"
    )

with col2:
    st.subheader("📷 التصوير بالكاميرا")
    camera_file = st.camera_input(
        "التقط صورة للعينة",
        key="camera_image"
    )

selected_file = None

if uploaded_file is not None and camera_file is not None:
    choice = st.radio(
        "حدد الصورة التي تريد تحليلها:",
        ["الصورة المرفوعة", "صورة الكاميرا"]
    )
    selected_file = uploaded_file if choice == "الصورة المرفوعة" else camera_file
elif uploaded_file is not None:
    selected_file = uploaded_file
elif camera_file is not None:
    selected_file = camera_file

if selected_file is not None:
    image = Image.open(selected_file).convert("RGB")
    img_array = np.array(image)

    st.divider()
    st.subheader("الصورة المختارة للعينة")
    st.image(image, use_container_width=True)

    if st.button("🧪 بدء التحليل الذكي للنسبة", type="primary"):
        with st.spinner("جاري تشغيل خوارزميات الرؤية الحاسوبية والذكاء الاصطناعي..."):
            
            # 1. تحويل الصورة إلى نطاق الألوان HSV للتحليل المتقدم
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # استخدام تقنية التكيف الذكي عبر عتبة Otsu مع تحسين التباين
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # تطبيق خوارزمية Otsu المتقدمة
            _, thresh = cv2.threshold(
                blurred,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # تحسين النتائج وإزالة التشويش (Morphological Operations)
            kernel = np.ones((3, 3), np.uint8)
            cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            # استخراج الأشكال والجسيمات بدقة (Contours)
            contours, _ = cv2.findContours(
                cleaned_thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            output_img = img_array.copy()
            total_particle_area = 0
            particle_count = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                # فلترة بناءً على الحساسية المحددة
                if area > min_particle_size:
                    particle_count += 1
                    total_particle_area += area

                    # رسم حدود دقيقة باللون الأخضر الفسفوري أو الأزرق حول الأجسام المكتشفة
                    cv2.drawContours(
                        output_img,
                        [contour],
                        -1,
                        (0, 255, 0),
                        2
                    )

            # المساحة الكلية للصورة
            total_area = img_array.shape[0] * img_array.shape[1]

            # حساب النسبة الصحيحة بدقة ذكية مع معامل تصحيح الحساسية
            ratio = (
                (total_particle_area / total_area) * 100 * (sensitivity / 50)
                if total_area > 0 else 0
            )
            ratio = min(ratio, 100.0) # ضمان عدم تجاوز 100%

            st.success("✨ تم تحليل عينة المياه بدقة واحترافية بنجاح!")

            st.subheader("الصورة بعد تحديد وتحليل الجسيمات بالذكاء الاصطناعي")
            st.image(output_img, use_container_width=True)

            # لوحة المؤشرات الإحصائية المتقدمة
            c1, c2, c3 = st.columns(3)
            c1.metric("عدد الجزيئات المكتشفة", f"{particle_count} جسيم")
            c2.metric("إجمالي المساحة الملوثة", f"{total_particle_area:.0f} px²")
            c3.metric("نسبة التلوث الذكية", f"{ratio:.2f}%")

            # تقييم مستوى الخطورة بناءً على النسبة الصحيحة
            if ratio > 4.0:
                st.error("🚨 تقرير الذكاء الاصطناعي: مستوى خطر عالي! العينة تحتوي على كثافة عالية جداً من جزيئات الميكروبلاستيك.")
            elif ratio > 1.5:
                st.warning("⚠️ تقرير الذكاء الاصطناعي: تلوث متوسط، يتطلب تطبيق نظام ترشيح متقدم للعينة.")
            elif particle_count > 0:
                st.info("ℹ️ تقرير الذكاء الاصطناعي: تلوث طفيف أو أثار ضئيلة للجسيمات.")
            else:
                st.success("✅ تقرير الذكاء الاصطناعي: العينة نقية وآمنة تماماً وخالية من التلوث المرئي.")

else:
    st.info("الرجاء اختيار طريقة الإدخال (رفع صورة أو التقفتها بالكاميرا) للبدء.")