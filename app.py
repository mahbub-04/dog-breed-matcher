import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ============================
#  Page configuration
# ============================

st.set_page_config(
    page_title="🐶 AI Dog Breed Matchmaker",
    page_icon="🐶",
    layout="centered"
)

# ============================
#  Data loading
# ============================

@st.cache_data
def load_data():
    breed_df = pd.read_csv("breed_traits.csv")
    trait_df = pd.read_csv("trait_description.csv")
    return breed_df, trait_df

breed_df, trait_df = load_data()

# ============================
#  NLP model loading
# ============================

@st.cache_resource
def load_model():
    model_name = "paraphrase-MiniLM-L6-v2"
    return SentenceTransformer(model_name)

model = load_model()

# ============================
#  Trait ideal phrases for NLP scoring
# ============================

trait_ideal_phrases = {
    "Affectionate With Family": [
        "Independent and distant.",
        "Sometimes needs family but mostly alone.",
        "Moderate bonding.",
        "Loves being close.",
        "Super affectionate."
    ],
    "Good With Young Children": [
        "Prefers adults only.",
        "Somewhat gentle.",
        "Neutral.",
        "Gentle and patient.",
        "Excellent with kids."
    ],
    "Good With Other Dogs": [
        "Avoids other dogs.",
        "Tolerates some.",
        "Neutral.",
        "Social with most.",
        "Loves meeting dogs."
    ],
    "Shedding Level": [
        "No shedding.",
        "Very low.",
        "Moderate.",
        "Fur sometimes.",
        "Heavy shed."
    ],
    "Coat Grooming Frequency": [
        "No grooming.",
        "Easy care.",
        "Some brushing.",
        "Needs regular.",
        "High-maintenance."
    ],
    "Trainability Level": [
        "Very stubborn.",
        "Some independence.",
        "Average.",
        "Quick learner.",
        "Super fast."
    ],
    "Energy Level": [
        "Couch potato.",
        "Chill.",
        "Active sometimes.",
        "Needs daily play.",
        "Super playful."
    ],
    "Mental Stimulation Needs": [
        "Low-key.",
        "Simple games.",
        "Enjoys puzzles.",
        "Wants challenges.",
        "Needs lots."
    ]
}

# ============================
#  NLP answer to score conversion
# ============================

def nlp_answer_to_score(answer: str, trait: str) -> int:
    """
    Convert a free-text answer into a 1–5 score for a given trait
    using sentence transformer embeddings and cosine similarity.
    """
    if not answer or answer.strip().lower() in ["skip", "idk", "not sure"]:
        # Neutral default in the middle of the scale
        return 3

    reference_sentences = trait_ideal_phrases[trait]
    embeddings = model.encode(reference_sentences + [answer], convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[-1], embeddings[:-1]).cpu().numpy().flatten()

    best_index = int(np.argmax(similarities))
    score = best_index + 1  # map index 0–4 to score 1–5

    return score

# ============================
#  Breed recommendation logic
# ============================

def get_breed_recommendations(trait_scores: dict, top_k: int = 3) -> pd.DataFrame:
    """
    Given a dictionary of {trait_name: user_score}, compute a match score
    for each breed and return the top_k best matches.
    """
    columns_used = list(trait_scores.keys())
    filtered = breed_df[["Breed"] + columns_used].copy()

    scores = []
    highlights = []

    for _, row in filtered.iterrows():
        total = 0
        highlight_row = []

        for trait in columns_used:
            diff = abs(int(row[trait]) - trait_scores[trait])

            # Higher is better: perfect match (diff 0) gives 5 points, then 4, 3, ...
            total += (5 - diff)

            if diff == 0:
                highlight_row.append("🟢")
            elif diff == 1:
                highlight_row.append("🟡")
            else:
                highlight_row.append("🔴")

        scores.append(total)
        highlights.append(highlight_row)

    filtered["Match Score"] = scores
    filtered["Highlights"] = highlights

    return (
        filtered
        .sort_values("Match Score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

# ============================
#  Main questions
# ============================

main_questions = [
    {
        "trait": "Affectionate With Family",
        "prompt": "🏠 Home moments: Snuggly pal or independent buddy?",
        "example": "(e.g., always together / likes space)"
    },
    {
        "trait": "Good With Young Children",
        "prompt": "👶 With kids: Gentle playmate or prefer adults for company?",
        "example": "(e.g., great with kids / adults only)"
    },
    {
        "trait": "Good With Other Dogs",
        "prompt": "🐕 At the park: Excited for new dog friends, or loyal to owner?",
        "example": "(e.g., social at park / shy with dogs)"
    },
    {
        "trait": "Shedding Level",
        "prompt": "🧹 Fur around house: No problem with fur, or need it neat?",
        "example": "(e.g., neat home / fur doesn't bother me)"
    },
    {
        "trait": "Coat Grooming Frequency",
        "prompt": "✂️ Grooming needs: Fine with daily care, or want easy coat?",
        "example": "(e.g., regular brushing / no fuss)"
    },
    {
        "trait": "Trainability Level",
        "prompt": "🎓 Training: Quick learner or adds own twists?",
        "example": "(e.g., fast learner / creative thinker)"
    },
    {
        "trait": "Energy Level",
        "prompt": "⚡ Daily life: Adventurer or chill companion?",
        "example": "(e.g., lots of play / prefers relaxing)"
    },
    {
        "trait": "Mental Stimulation Needs",
        "prompt": "🧠 Brain games: Loves new puzzles, or low key?",
        "example": "(e.g., puzzle games / laid-back)"
    },
]

# ============================
#  UI: Title and introduction
# ============================

st.title("🐶 AI Dog Breed Matchmaker")
st.markdown(
    "Answer a few natural-language questions about your lifestyle and preferences, "
    "and this AI assistant will suggest the top 3 dog breeds that best fit you."
)
st.markdown("---")

# ============================
#  Conversation state
# ============================

if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.responses = {}

# ============================
#  Question flow
# ============================

if st.session_state.step < len(main_questions):
    q = main_questions[st.session_state.step]

    st.subheader(f"Question {st.session_state.step + 1} of {len(main_questions)}")
    st.write(f"**{q['prompt']}**")
    st.caption(q["example"])

    user_input = st.text_input("Your answer:", key=f"input_{st.session_state.step}")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Next ➡️", key="next"):
            if user_input.strip():
                st.session_state.responses[q["trait"]] = user_input
                st.session_state.step += 1
                st.rerun()
            else:
                st.warning("Please enter an answer or type 'skip'.")

    with col2:
        if st.button("Skip", key="skip"):
            st.session_state.responses[q["trait"]] = "skip"
            st.session_state.step += 1
            st.rerun()

# ============================
#  Results view
# ============================

else:
    st.success("✅ All questions answered. Processing your results...")

    # Convert answers to scores using NLP
    trait_scores = {}
    for trait, answer in st.session_state.responses.items():
        trait_scores[trait] = nlp_answer_to_score(answer, trait)

    st.markdown("### 🎯 Your Trait Preferences")
    for trait, score in trait_scores.items():
        st.write(f"- **{trait}**: {score}/5")

    st.markdown("---")
    st.markdown("### 🏆 Top 3 Breed Matches")

    # Get recommendations
    top_breeds = get_breed_recommendations(trait_scores, top_k=3)

    for idx, row in top_breeds.iterrows():
        with st.expander(
            f"#{idx + 1} **{row['Breed']}** – Match Score: {row['Match Score']}",
            expanded=(idx == 0)
        ):
            st.write(f"**Match Highlights:** {' '.join(row['Highlights'])}")

            cols = st.columns(len(trait_scores))
            for i, (trait, score) in enumerate(trait_scores.items()):
                with cols[i]:
                    st.metric(trait.replace(" ", "\n"), f"{int(row[trait])}/5")

    st.markdown("---")
    if st.button("🔄 Start Over"):
        st.session_state.step = 0
        st.session_state.responses = {}
        st.rerun()
