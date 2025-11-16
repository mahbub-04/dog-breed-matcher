# AI Dog Breed Matchmaker

An AI-powered dog breed matchmaker that recommends the top three real-world breeds based on a user's lifestyle and personality.  
This project was developed for the “Choose My Dog Breed: AI Chatbot” innovation challenge and is designed to be both competition-ready and portfolio-ready.

---

## Project Overview

Choosing a dog is effectively a long-term lifestyle decision, but many new owners still choose based on looks, trends, or vague online quizzes.  
This project connects free-text user input to structured breed traits so that people can find breeds that actually fit their day-to-day life.

The system:

- Asks a short sequence of open-ended, conversational questions instead of rigid multiple-choice forms.  
- Uses a sentence-transformer model to embed user answers and map them to trait scores on a 1–5 scale.  
- Computes a data-driven match score between the user profile and each breed in a curated trait dataset.  
- Returns the top three matching breeds, with simple visual indicators that explain how well each trait aligns.

---

## Key Features

- Natural-language chatbot experience  
  Users answer in free text (for example: “I live in an apartment and enjoy evening walks”), and the system interprets the meaning.

- Embedding-based trait scoring  
  A SentenceTransformer model (`paraphrase-MiniLM-L6-v2`) maps each answer to traits such as:  
  Affectionate With Family, Good With Young Children, Good With Other Dogs, Shedding Level, Coat Grooming Frequency, Trainability Level, Energy Level, and Mental Stimulation Needs.

- Transparent matching logic  
  Breed compatibility is calculated using an explicit, interpretable formula based on the difference between user trait scores and breed trait scores.

- Top-three recommendations with highlights  
  Each recommended breed is displayed with a match score and simple indicators (for example, green, yellow, red) that show where the match is strong or weaker.

- Deployed web application  
  The full experience is available as a Streamlit app, alongside a reproducible notebook for detailed inspection.

---

## Data

This project uses two main CSV files:

- `breed_traits.csv`  
  - One row per dog breed.  
  - Trait scores (1–5) for characteristics such as Affectionate With Family, Good With Young Children, Good With Other Dogs, Shedding Level, Coat Grooming Frequency, Trainability Level, Energy Level, and Mental Stimulation Needs.

- `trait_description.csv`  
  - Descriptions of each trait and what different scores (1–5) mean in practice.  
  - Used to keep the numeric scale aligned with human-readable explanations.

These datasets are used both in the notebook (for analysis and explanation) and in the Streamlit app (for live recommendations).

---

## Core Matching Logic

1. Convert answers to trait scores  

   For each trait (for example, Energy Level), the system defines five short reference phrases representing scores from 1 to 5.  
   A SentenceTransformer model embeds these phrases and the user’s answer, and cosine similarity is used to find the closest phrase, which maps to a score between 1 and 5.

2. Compute a match score for each breed  

   For each breed and each trait used in the conversation:

   \[
   \text{points} = 5 - \lvert \text{user\_score} - \text{breed\_score} \rvert
   \]

   The total match score for a breed is the sum of these points across all traits.  
   Higher scores indicate better alignment between the user’s lifestyle and the breed’s characteristics.

3. Return the top three breeds  

   All breeds are sorted by match score, and the top three are returned with their scores and per-trait indicators.

---

## Live Demo

The chatbot is deployed as a Streamlit application:

- Streamlit app: https://dog-breed-matcher-mash.streamlit.app

The app:

- Asks eight open-ended questions mapped to key lifestyle traits.  
- Shows the interpreted trait profile for the user.  
- Displays the top three recommended breeds with match scores and per-trait metrics.

---

## Notebook

The full reasoning, data exploration, and algorithm explanation are documented in:

- `Choose-My-Dog-Breed_Competition_Submission.ipynb`

The notebook is structured as:

1. Executive summary and problem framing  
2. Dataset loading and exploratory analysis  
3. NLP-based trait scoring with sentence embeddings  
4. Breed matching logic and example runs  
5. User journey and alignment with competition judging criteria  
6. Reproducibility instructions

This structure makes it easy for judges, recruiters, or collaborators to review both the implementation and the narrative.

---

## How to Run Locally

### 1. Clone the repository

### 2. Install dependencies

### 3. Run the Streamlit app

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

### 4. Run the notebook (optional)

- Open `Choose-My-Dog-Breed_Competition_Submission.ipynb` in Jupyter, Colab, or Vertex AI Workbench / Datalab.  
- Ensure `breed_traits.csv` and `trait_description.csv` are in the same directory.  
- Run all cells from top to bottom.

---

## Tech Stack

- Frontend / App: Streamlit  
- Language: Python  
- NLP: SentenceTransformers (`paraphrase-MiniLM-L6-v2`)  
- Data and analysis: pandas, numpy, matplotlib, seaborn  
- Deployment: Streamlit Cloud (connected to this GitHub repository)

---

## Author

- Name: Md. Mahbub-Ur-Rashid  
- Role: Data science / machine learning enthusiast  
- Contact: mash12mahbub@gmail.com



