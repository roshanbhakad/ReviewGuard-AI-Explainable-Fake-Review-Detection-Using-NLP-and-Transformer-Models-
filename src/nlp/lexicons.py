"""Domain lexicons, sentiment dictionaries, and deceptive review markers.
Built-in zero-dependency lexicons based on empirical deceptive spam research.
"""

# VADER-inspired sentiment polarity lexicon (subset of validated valence words)
SENTIMENT_LEXICON = {
    # Strong positive (+3.0 to +4.0)
    "excellent": 3.2, "exceptional": 3.4, "masterpiece": 3.5, "perfection": 3.6,
    "outstanding": 3.3, "flawless": 3.5, "superb": 3.2, "magnificent": 3.4,
    "phenomenal": 3.6, "brilliant": 3.1, "spectacular": 3.3, "breathtaking": 3.4,
    "delightful": 2.8, "wonderful": 2.9, "fantastic": 3.0, "amazing": 3.0,
    "love": 2.8, "loved": 2.9, "loves": 2.8, "loving": 2.6,
    "awesome": 2.8, "great": 2.3, "good": 1.9, "best": 3.0, "better": 1.5,
    "happy": 2.2, "pleased": 2.1, "recommend": 2.2, "recommended": 2.3,
    "satisfaction": 2.2, "satisfied": 2.1, "worth": 1.8, "helpful": 1.9,
    "reliable": 2.1, "durable": 2.0, "fast": 1.5, "smooth": 1.6,
    "clean": 1.8, "cozy": 2.0, "friendly": 2.2, "polite": 1.9,
    "delicious": 2.7, "tasty": 2.2, "fresh": 1.8, "favorite": 2.4,
    "impressed": 2.5, "stunning": 3.1, "ideal": 2.4, "solid": 1.7,

    # Moderate positive (+1.0 to +2.0)
    "decent": 1.2, "fine": 1.1, "okay": 0.8, "acceptable": 1.0, "fair": 0.9,
    "nice": 1.5, "neat": 1.3, "pleasant": 1.8, "usable": 1.2, "works": 1.1,

    # Strong negative (-3.0 to -4.0)
    "terrible": -3.2, "horrible": -3.4, "atrocious": -3.6, "abysmal": -3.5,
    "disaster": -3.3, "disastrous": -3.4, "catastrophe": -3.5, "scam": -3.7,
    "fraud": -3.8, "rip-off": -3.5, "ripoff": -3.5, "garbage": -3.3,
    "trash": -3.2, "junk": -3.0, "pathetic": -3.2, "worthless": -3.4,
    "unacceptable": -3.1, "appalling": -3.5, "worst": -3.5, "nightmare": -3.4,
    "hate": -3.0, "hated": -3.1, "awful": -3.1, "ruined": -3.0,
    "useless": -3.0, "fake": -3.2, "broken": -2.8, "defective": -2.9,

    # Moderate negative (-1.0 to -2.5)
    "bad": -2.1, "poor": -2.2, "disappointed": -2.3, "disappointing": -2.4,
    "slow": -1.5, "noisy": -1.4, "dirty": -2.2, "rude": -2.5, "smelly": -2.0,
    "buggy": -2.1, "flimsy": -2.3, "expensive": -1.4, "overpriced": -2.2,
    "regret": -2.3, "avoid": -2.6, "fail": -2.4, "failed": -2.5, "fails": -2.4,
    "waste": -2.8, "cheap": -1.8, "problem": -1.7, "problems": -1.8, "issue": -1.5,
    "issues": -1.6, "unhappy": -2.1, "annoying": -2.2, "annoyed": -2.0,
}

# Negation words that invert polarity
NEGATION_WORDS = {
    "not", "never", "no", "neither", "nor", "hardly", "barely",
    "scarcely", "seldom", "rarely", "without", "cannot", "cant",
    "can't", "don't", "dont", "didn't", "didnt", "won't", "wont",
    "wouldn't", "wouldnt", "shouldn't", "shouldnt", "isn't", "isnt",
    "wasn't", "wasnt", "aren't", "arent", "weren't", "werent"
}

# Polarity booster words
BOOSTER_WORDS = {
    "absolutely": 1.5, "extremely": 1.5, "definitely": 1.3,
    "incredibly": 1.5, "completely": 1.4, "totally": 1.4,
    "highly": 1.3, "really": 1.3, "super": 1.3, "very": 1.25,
    "extra": 1.25, "insanely": 1.5, "unquestionably": 1.5,
    "somewhat": 0.7, "slightly": 0.6, "barely": 0.5, "kinda": 0.7
}

# Deceptive spam cues and promotional markers
SPAM_PROMOTIONAL_MARKERS = {
    "must buy", "best product ever", "changed my life", "waste of money",
    "100% recommend", "do not buy", "run away", "scam alert", "guaranteed results",
    "free sample", "paid review", "gifted", "sponsored", "honest review in exchange",
    "click the link", "discount code", "promo code", "five stars all the way",
    "hands down the best", "miracle product", "holy grail", "cannot live without",
    "best purchase ever", "look no further", "buy it now", "worth every single penny",
    "do yourself a favor", "run don't walk", "runs don't walk", "life changer",
    "take my word", "trust me", "simply the best", "beyond expectations"
}

# Hyperbolic words frequently overrepresented in synthetic/fake reviews
HYPERBOLIC_WORDS = {
    "miracle", "magical", "flawless", "perfection", "unbelievable",
    "astounding", "revolutionary", "godsend", "spectacular", "phenomenal",
    "breathtaking", "unrivaled", "incomparable", "transcendent",
    "legendary", "gamechanger", "game-changer", "masterpiece", "unmatched"
}

# Pronoun categories (Ott et al. deceptive opinion spam stylometrics)
PRONOUNS_FIRST_SINGULAR = {"i", "me", "my", "mine", "myself"}
PRONOUNS_FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}
PRONOUNS_SECOND = {"you", "your", "yours", "yourself", "yourselves"}
PRONOUNS_THIRD = {"he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"}

# Common English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves"
}
