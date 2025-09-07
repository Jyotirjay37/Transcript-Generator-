# 📝 Real-time Meeting Summarizer (Free & Open Source)

Welcome to the **Meeting Summarizer** project! This app helps you transcribe, summarize, and extract key insights from your meetings — all using free and open-source models. Perfect for teams who want quick, actionable meeting notes without relying on paid services.

---

## 🚀 Project Highlights

- 🎤 **Transcription:** Uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) for fast, local speech-to-text transcription.
- 🧠 **Summarization:** Powered by Hugging Face's DistilBART model for concise, two-pass meeting summaries.
- 📋 **Action Items & Decisions:** Extracts key decisions, action items, and risks using keyword heuristics and optional FLAN-T5 model.
- 🖥️ **User Interface:** Built with [Streamlit](https://streamlit.io/) for an easy-to-use web app experience.
- 🔒 **Privacy:** Runs fully locally on your machine — no data leaves your computer.
- ⚙️ **Customizable:** Choose model sizes, precision, and summarization settings to balance speed and accuracy.

---

## 🎯 What You Get

- **Executive Summary:** A clear, concise overview of your meeting.
- **Structured Minutes:** Lists of key decisions, action items, and risks/blockers.
- **Transcript Preview:** View the full transcript or download it as an SRT subtitle file.
- **Downloadable Outputs:** Export meeting minutes in Markdown and transcripts in SRT format for easy sharing and archiving.

---

## 🛠️ How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:

   ```bash
   streamlit run music.py
   ```

3. Upload your audio/video file and click **Run Summarization**.

---

## 📂 Supported File Types

- Audio: WAV, MP3, M4A
- Video: MP4, MKV, MOV

---

## 💡 Tips

- Start with smaller ASR and summarization models for faster results.
- Enable FLAN-T5 for more detailed action item extraction (may be slower).
- Use the sidebar settings to customize your experience.

---

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the project!

---

## 📄 License

This project is open source and free to use.

---

Thank you for using the Meeting Summarizer! 🎉
