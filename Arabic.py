!pip install -q transformers gradio torch --upgrade

import gradio as gr
from transformers import pipeline
import torch

# Choosing 'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'
# Reason: It is trained on the CAMeL-Lab dataset which covers MSA and multiple dialects (DA),
# making it highly effective for real-world social media comments.
model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Load the sentiment analysis pipeline
# تحميل أنبوب تحليل المشاعر
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name,
    device=device
)

def analyze_sentiment(text):
    """
    Function to predict sentiment and return formatted HTML/Markdown.
    دالة لتوقع المشاعر وإرجاع النتيجة بتنسيق مناسب.
    """
    if not text.strip():
        return "⚠️ الرجاء إدخال نص للتحليل | Please enter text to analyze."

    try:
        results = sentiment_pipeline(text)
        label = results[0]['label']
        score = results[0]['score'] * 100

        # Mapping labels to Arabic and Emojis
        # تحويل التصنيفات إلى اللغة العربية والرموز التعبيرية
        sentiment_map = {
            "positive": ("إيجابي", "👍"),
            "negative": ("سلبي", "👎"),
            "neutral": ("محايد", "😐")
        }

        # Standardizing label format (some models return 'LABEL_1' etc, but CAMeL uses lower strings)
        ar_label, emoji = sentiment_map.get(label.lower(), (label, ""))

        color = "#27ae60" if label.lower() == "positive" else "#c0392b" if label.lower() == "negative" else "#7f8c8d"

        result_html = f"""
        <div style='text-align: right; direction: rtl; padding: 20px; border-radius: 10px; background-color: {color}; color: white;'>
            <h2 style='margin: 0;'>النتيجة: {ar_label} {emoji}</h2>
            <p style='font-size: 1.2em;'>ثقة النموذج: {score:.2f}%</p>
        </div>
        """
        return result_html

    except Exception as e:
        return f"Error: {str(e)}"

# Define Examples (MSA, Egyptian, Gulf, Levantine)
# أمثلة متنوعة (فصحى، مصري، خليجي، شامي)
examples = [
    ["الكتاب كان مفيداً جداً واستمتعت بقراءته."], # MSA - Positive
    ["الفيلم ده وحش قوي وضياع للوقت بصراحة."], # Egyptian - Negative
    ["يا زين هالأخبار، الله يسعدكم دوم."], # Gulf - Positive
    ["شو هالمعاملة السيئة؟ ما بنصح حدا يروح لهنيك."], # Levantine - Negative
    ["الجو اليوم في دبي وايد حر بس جميل."], # Gulf - Mixed/Positive
    ["بصراحة الخدمة نص نص، مش بطالة بس محتاجة تحسين."], # Egyptian - Neutral/Mixed
    ["أنا فخور جداً بالإنجازات العربية في مجال العلوم."], # MSA - Positive
    ["تأخير بالطلب وتغليف سيء جداً."] # MSA/General - Negative
]

# Create Gradio Interface
# إنشاء واجهة جراديو
with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {direction: rtl}") as demo:
    gr.Markdown("# <center>تحليل المشاعر العربية - Arabic Sentiment Analysis</center>")
    gr.Markdown("""
    <center>
    أهلاً بك! استخدم هذا النموذج لتحليل مشاعر النصوص العربية (فصحى ولهجات).
    Welcome! Use this model to analyze the sentiment of Arabic text (MSA and Dialects).
    </center>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="أدخل النص هنا / Enter Text",
                placeholder="مثال: المنتج رائع جداً...",
                lines=5
            )
            submit_btn = gr.Button("تحليل / Analyze", variant="primary")

        with gr.Column():
            output_html = gr.HTML(label="النتيجة / Result")

    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=output_html,
        fn=analyze_sentiment,
        cache_examples=False
    )

    submit_btn.click(fn=analyze_sentiment, inputs=input_text, outputs=output_html)

    gr.Markdown("""
    ### 🚀 Ideas for Next Steps / أفكار للتطوير المستقبلي:
    1. **Fine-tuning**: Train on specific domain data like E-commerce reviews or Twitter data.
    2. **Neutral Class**: Some models handle 3 classes (Pos/Neg/Neu); adding Neutral improves accuracy for objective news.
    3. **Deployment**: Deploy to **Hugging Face Spaces** for a permanent URL.
    4. **Batch Processing**: Add functionality to upload CSV/Excel files and analyze thousands of comments at once.
    """)

# Launch with share=True for Colab
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
