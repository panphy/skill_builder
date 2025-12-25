PanPhy Skill Builder

PanPhy Skill Builder is a Streamlit-based prototype that helps GCSE Physics students practise short exam-style questions with instant, structured feedback. It is designed to support skill-building and good exam habits by encouraging students to show working, then receiving mark-style comments and clear next steps.

The app offers two question sources:
	•	Built-in practice questions for quick testing and demos
	•	Teacher Uploads (custom question bank) where teachers can upload a question screenshot and the corresponding mark scheme screenshot (one question at a time). These are stored in Supabase Storage with metadata in Supabase Postgres, so they can be reused later.

Students can respond in two ways:
	•	Type Answer (working in a text box)
	•	Write Answer (handwrite working on a canvas, suitable for iPad use)

For each submission, the app uses an OpenAI model (currently gpt-5-mini) to generate a JSON-only marking report containing:
	•	marks awarded (out of the max)
	•	a short summary
	•	specific feedback points
	•	actionable next steps

Teachers can access a password-protected dashboard to view class performance and attempt history, including:
	•	total attempts, unique students, topics attempted
	•	performance by student and by topic
	•	recent attempts feed

This repo contains the current Streamlit prototype and the supporting database/storage integration, and serves as the foundation for a future hybrid version where the student writing experience can be made smoother using a dedicated web front end.