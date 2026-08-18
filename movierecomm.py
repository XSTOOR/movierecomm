import streamlit as st
import pandas as pd

# ------------------------------------------------------------------
# Page config & theme
# ------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading movie library…")
def load_data():
    df = pd.read_csv("movies.csv")
    # Normalise the columns we actually use so the UI never trips on NaNs.
    for col in ["title", "genres", "overview", "tagline", "director", "cast"]:
        df[col] = df[col].fillna("")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    # Deduplicate by title, keeping the most popular entry so the picker is clean.
    df = df.sort_values("popularity", ascending=False).drop_duplicates("title").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def all_genres(df: pd.DataFrame):
    """Return a sorted list of every genre present in the dataset."""
    genres = set()
    for g in df["genres"].dropna():
        genres.update(g.split())
    return sorted(genres)


movies_df = load_data()
GENRES = all_genres(movies_df)


# ------------------------------------------------------------------
# Similarity & recommendations
# ------------------------------------------------------------------
def parse_genres(genres_str: str) -> set:
    """Genres in this dataset are space-separated (e.g. 'Action Adventure')."""
    if not isinstance(genres_str, str) or not genres_str.strip():
        return set()
    return set(genres_str.split())


def calculate_similarity(genres_1: str, genres_2: str) -> float:
    """Jaccard similarity over the genre sets of two movies."""
    g1 = parse_genres(genres_1)
    g2 = parse_genres(genres_2)
    if not g1 or not g2:
        return 0.0
    intersection = g1 & g2
    union = g1 | g2
    return len(intersection) / len(union)


@st.cache_data(show_spinner="Finding similar movies…")
def get_recommendations(movie_title: str, threshold: float = 0.2, top_n: int = 10):
    """Return the most genre-similar movies to the given title.

    Results are sorted by similarity (then by popularity as a tie-breaker)
    and returned as a DataFrame with a `similarity` column.
    """
    matches = movies_df[movies_df["title"] == movie_title]
    if matches.empty:
        return pd.DataFrame()
    movie_genres = matches["genres"].iloc[0]

    others = movies_df[movies_df["title"] != movie_title].copy()
    others["similarity"] = others["genres"].apply(lambda g: calculate_similarity(movie_genres, g))
    recs = others[others["similarity"] >= threshold]
    recs = recs.sort_values(["similarity", "popularity"], ascending=[False, False]).head(top_n)
    return recs.reset_index(drop=True)


def movie_lookup(title: str):
    row = movies_df[movies_df["title"] == title]
    return row.iloc[0] if not row.empty else None


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------
def star_rating(score: float) -> str:
    """Render a 5-star rating string from a 0-10 score."""
    if score is None or pd.isna(score):
        return "—"
    stars = round(score / 2)
    return "★" * stars + "☆" * (5 - stars)


def genre_badges(genres_str: str):
    """Render genre tags as coloured pills."""
    genres = parse_genres(genres_str)
    if not genres:
        st.caption("_No genres listed_")
        return
    cols = st.columns(len(genres))
    for col, genre in zip(cols, genres):
        col.markdown(
            f"<span style='background-color:#e50914;color:#fff;padding:4px 10px;"
            f"border-radius:12px;font-size:0.78rem;font-weight:600;'>{genre}</span>",
            unsafe_allow_html=True,
        )


def format_runtime(minutes) -> str:
    if pd.isna(minutes) or minutes <= 0:
        return "—"
    h, m = divmod(int(minutes), 60)
    return f"{h}h {m}m" if h else f"{m}m"


def format_year(row) -> str:
    year = row.get("year")
    if pd.isna(year):
        return "—"
    return str(int(year))


def show_movie_card(row: pd.Series, rank: int | None = None, similarity: float | None = None):
    """Render a single movie as a compact, attractive card."""
    prefix = f"#{rank} " if rank else ""
    title_line = f"{prefix}**{row['title']}**"
    if similarity is not None:
        title_line += f"  ·  match `{similarity:.0%}`"

    with st.container(border=True):
        st.markdown(title_line)
        meta_bits = [
            f"⭐ {row['vote_average']:.1f}/10" if row["vote_average"] else None,
            f"🗓 {format_year(row)}",
            f"⏱ {format_runtime(row['runtime'])}",
            f"👥 {int(row['vote_count']):,} votes" if row["vote_count"] else None,
        ]
        meta = "  ·  ".join(b for b in meta_bits if b)
        if meta:
            st.caption(meta)
        genre_badges(row["genres"])
        if row["tagline"]:
            st.markdown(f"_{row['tagline']}_")
        if row["overview"]:
            st.markdown(row["overview"][:300] + ("…" if len(row["overview"]) > 300 else ""))
        cast = str(row.get("cast", "")).split()
        if cast:
            st.caption("🎬 Cast: " + " ".join(cast[:6]))
        if row.get("director"):
            st.caption(f"🎥 Director: {row['director']}")


# ------------------------------------------------------------------
# Custom CSS for a cinematic look
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark cinematic theme */
    .stApp {
        background: linear-gradient(160deg, #141414 0%, #1b1b1b 60%, #0f0f0f 100%);
    }
    /* Tighten section headers */
    .stApp h1, .stApp h2, .stApp h3 { letter-spacing: 0.3px; }
    /* Cards / containers */
    .stVerticalBlock > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px;
        transition: transform .15s ease, border-color .15s ease;
    }
    .stVerticalBlock > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        border-color: rgba(229,9,20,0.5) !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #e50914;
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #f6121d;
        color: #fff;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(0,0,0,0.35);
    }
    /* Caption spacing */
    .stApp .stCaption { line-height: 1.4; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    "<h1 style='color:#e50914;'>🎬 CineMatch</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Find your next favorite film. Pick a movie you love and we'll surface "
    "genre-similar titles from a library of **{:,}** movies.".format(len(movies_df))
)
st.divider()

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🎛 Controls")
    selected_movie = st.selectbox(
        "Choose a movie",
        movies_df["title"].sort_values().tolist(),
        index=0,
        help="Start typing to search the full library.",
    )
    threshold = st.slider(
        "Minimum match strength",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Higher = only very similar genres. Lower = broader recommendations.",
    )
    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
    )
    st.divider()
    if st.button("🎲 Surprise me!", use_container_width=True):
        selected_movie = movies_df.sample(1)["title"].iloc[0]
        st.toast(f"Selected: {selected_movie}", icon="🎲")
    st.caption("Tip: lower the match strength if you get too few results.")

# ------------------------------------------------------------------
# Selected movie spotlight
# ------------------------------------------------------------------
spotlight = movie_lookup(selected_movie)
if spotlight is not None:
    st.subheader("Now showing")
    show_movie_card(spotlight)
    st.divider()

# ------------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------------
st.subheader("Recommended for you")

recs = get_recommendations(selected_movie, threshold=threshold, top_n=top_n)

if recs.empty:
    st.info(
        "No recommendations found at this match strength. "
        "Try lowering the **Minimum match strength** in the sidebar."
    )
else:
    st.caption(f"Showing {len(recs)} of the closest matches to **{selected_movie}**.")
    for i, (_, row) in enumerate(recs.iterrows(), start=1):
        show_movie_card(row, rank=i, similarity=row["similarity"])

st.divider()
st.caption("CineMatch · genre-based recommendations · data from TMDB")
