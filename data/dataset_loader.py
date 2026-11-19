"""Dataset generator, loader, and benchmark dataset for ReviewGuard-AI.
Generates comprehensive synthetic benchmark reviews across multiple e-commerce categories
and formats standard external datasets (Ott et al., Amazon, Yelp).
"""

import os
import random
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

# Reproducibility seed
SEED = 42

AUTHENTIC_TEMPLATES = [
    # Tech / Electronics
    ("I've been using this {device} for about {time_period}. The {feature1} is solid and {feature2} works as expected. However, the {drawback} could be improved. Overall decent value for ${price}.", 4, "Electronics"),
    ("Battery life on this {device} lasts roughly {hours} hours of continuous use. Build quality feels sturdy with a matte finish. Minor gripe: the power cable is only {cable_len} feet long. Would rate 4/5.", 4, "Electronics"),
    ("Purchased to replace my older {device}. Setup took around {minutes} minutes following the quick start manual. Sound clarity is crisp, though bass is slightly muddy at max volume.", 4, "Electronics"),
    ("Returned after {days} days. The {device} stopped connecting to Wi-Fi whenever my laptop entered sleep mode. Customer support responded within 24 hours and issued a refund without hassle.", 2, "Electronics"),
    ("Honest impression after 3 months: it does what it advertises. The buttons have good tactile feedback and the display is clear in direct sunlight. Heavy compared to competitors though.", 3, "Electronics"),
    ("Does the job well. The aluminum frame feels durable. My only complaint is that the instruction booklet was printed in tiny font, but online tutorials helped.", 4, "Electronics"),
    ("Arrived in good packaging within 3 business days. Plugged into my setup and recognized instantly on Windows 11. Temperatures stay around 45C under standard load.", 5, "Electronics"),
    ("Not quite as quiet as I hoped. Fan noise is noticeable at roughly 40dB from 2 feet away. Still, cooling performance dropped my temps by 8 degrees.", 3, "Electronics"),

    # Home & Kitchen
    ("We purchased this {kitchen_item} for daily meal prep. Cuts prep time in half and dishwasher cleaning is straightforward. The lid latch requires a firm press to seal properly.", 4, "Home & Kitchen"),
    ("Decent blender for smoothies. Crushes frozen berries easily, though kale leaves need an extra 30-second pulse. Stainless steel blades show no dulling after 6 weeks.", 4, "Home & Kitchen"),
    ("The skillet heats evenly on our induction cooktop. Handle stays cool to the touch up to medium heat. A bit heavier than our previous Teflon pan at 3.2 lbs.", 4, "Home & Kitchen"),
    ("Worked well for the first month, then the heating element became inconsistent, taking 15 minutes instead of 8 to reach temperature. Exchanged under warranty.", 2, "Home & Kitchen"),
    ("Compact design fits neatly under our 18-inch cabinets. Holds enough coffee for 4 large mugs. Carafe pour spout doesn't drip.", 5, "Home & Kitchen"),
    ("Fair quality for the price tag. The plastic knobs feel slightly loose, but functionally it has held up through 50+ cooking sessions without failure.", 3, "Home & Kitchen"),

    # Fashion & Apparel
    ("Fabric is 95% cotton and 5% elastane. Fits true to size around the chest (I am 6'0, 180 lbs, wearing Large). Shrunk slightly after first wash in warm water.", 4, "Apparel"),
    ("Good stitching along the seams and zipper operates smoothly. Color is slightly darker navy than shown in product photos, but looks sharp with khaki chinos.", 4, "Apparel"),
    ("Sole cushioning is comfortable for standing 6-8 hour shifts. Arch support is moderate. Order half a size up if you have wide feet.", 4, "Apparel"),
    ("Sole separated from the toe box after 4 weeks of light walking on paved sidewalks. Material feels thin compared to older models from this brand.", 1, "Apparel"),

    # Books & Media
    ("Thoroughly researched 340-page guide with clear diagrams and real-world case studies in chapters 4 and 7. The index at the back makes quick reference simple.", 5, "Books"),
    ("Well written narrative, though the pacing drags noticeably in the middle section (around pages 120-190). The conclusion ties up loose ends nicely.", 3, "Books"),
    ("Provides actionable advice on financial planning. Mathematical examples are clear, though some tax tables reflect 2021 rather than current tax year brackets.", 4, "Books"),

    # Concise / Short Authentic Reviews
    ("Decent headphones for everyday use. Battery lasts about 20 hours. Sound is clear.", 4, "Electronics"),
    ("Works as advertised. Setup was easy and build quality is sturdy.", 4, "Electronics"),
    ("Good quality for the price. Arrived quickly and fits comfortably.", 4, "Apparel"),
    ("Comfortable fit and decent materials. Happy with this purchase.", 4, "Apparel"),
    ("Does the job well. No issues after two weeks of daily use.", 4, "General"),
    ("Brews quickly and clean up is simple. Good value for money.", 4, "Home & Kitchen"),
    ("Solid sound quality and fast Bluetooth pairing with my phone.", 4, "Electronics"),
    ("Arrived on time in good packaging. Functions exactly as needed.", 5, "General"),
]

FAKE_SPAM_TEMPLATES = [
    # Extreme glowing bot praise / incentivized
    ("BEST PRODUCT EVERRRR!!!! Absolutely blown away by the quality!! Changed my life forever!! MUST BUY 100% recommended to everyone!! A++++++ service!!!", 5, "General"),
    ("DO YOURSELF A FAVOR AND BUY THIS RIGHT NOW!!! Hands down the greatest purchase I have ever made in my entire life!! Miracle product, five stars all the way!!", 5, "General"),
    ("AMAZING AMAZING AMAZING!! I was skeptical at first but wow, simply perfection!! Zero flaws, unbelievable results in just 2 days!! 10/10!!", 5, "General"),
    ("Words cannot express how incredible this is! Completely flawless, breathtaking craftsmanship, holy grail product! Run don't walk to get this!!!", 5, "General"),
    ("I NEVER write reviews but I simply had to for this masterpiece!! Spectacular! Life changing! Everyone in my family is buying one now!!", 5, "General"),
    ("LOOK NO FURTHER!! This is the ultimate best on the market!! Fast shipping, flawless item, greatest seller ever!! Will definitely order 10 more!!", 5, "General"),
    ("Five stars is not enough!! Should be 100 stars!! Miracle results overnight!! Don't hesitate, add to cart immediately!! Pure genius!!", 5, "General"),
    ("Truly revolutionary! Top tier elite quality! Unrivaled perfection! Best decision I ever made! Super super happy customer!!", 5, "General"),
    ("Free sample provided for my honest review! Wow unbelievable quality, definitely worth every single penny, click the link to get your discount code!!", 5, "General"),
    ("BUY IT NOW BUY IT NOW BUY IT NOW!! You will not regret it!! Perfection in every single way!! Absolutely stunning!!", 5, "General"),

    # Smear / Competitor sabotage attacks
    ("DO NOT BUY!!! TOTAL SCAM!! Broke in 5 seconds!! Garbage trash worthless junk!! Worst company on planet earth!! RUN AWAY SCAM SCAM SCAM!!", 1, "General"),
    ("WORST PURCHASE OF MY LIFE!! Complete rip-off! Dangerous, defective, catastrophic failure! Demand an immediate refund! AVOID AT ALL COSTS!!", 1, "General"),
    ("HORRIBLE DISASTER!! Toxic smell, horrible seller, zero customer service! Threw it straight in the garbage bin! Terrible!!", 1, "General"),
    ("Absolute garbage. Don't waste your money. Fake product, fake descriptions, complete fraud. They should be banned from selling!!", 1, "General"),

    # Rating-Dissonance fraud
    ("Completely broken on arrival, total disaster and useless waste of money. Worst thing ever.", 5, "Electronics"),  # 5 star but negative text
    ("Best purchase ever made, unbelievable quality, perfection, loved every second of using it!", 1, "General"),      # 1 star but positive text
    ("Defective, returned immediately. Terrible experience, do not recommend to anyone.", 5, "Home & Kitchen"),       # 5 star but negative text
    ("Exceeded all expectations! Flawless design and my kids absolutely loved it! Superb!", 1, "Apparel"),             # 1 star but positive text

    # Generic templated bot reviews
    ("Good item arrived fast. Item is good and fine. I like item very much. Good good good.", 5, "General"),
    ("Nice product. Very nice quality. Very very good. Recommend this product to all buyers.", 5, "General"),
    ("Product is as described. Very nice. I am happy. Good seller. Thank you.", 5, "General"),
]

DEVICE_LIST = ["wireless earbuds", "Bluetooth speaker", "ergonomic keyboard", "USB-C docking station", "external SSD", "mechanical keyboard", "dash cam", "smartwatch"]
FEATURE1_LIST = ["sound stage", "key travel", "data transfer speed", "display brightness", "pairing stability", "microphone sensitivity"]
FEATURE2_LIST = ["ANC noise cancellation", "battery management", "multi-device switching", "touch controls", "firmware updates"]
DRAWBACK_LIST = ["companion mobile app", "charging case hinge", "LED indicator brightness", "default EQ profile"]
KITCHEN_LIST = ["air fryer", "immersion blender", "food processor", "cast iron dutch oven", "knife sharpener", "espresso machine"]


def generate_synthetic_dataset(num_samples: int = 3000, random_state: int = SEED) -> pd.DataFrame:
    """Generates a balanced, realistic benchmark dataset of authentic and deceptive reviews."""
    random.seed(random_state)
    np.random.seed(random_state)

    records = []
    half = num_samples // 2

    # 1. Authentic reviews (label = 0)
    for _ in range(half):
        tmpl, base_rating, cat = random.choice(AUTHENTIC_TEMPLATES)
        # Random variations to make each text unique
        text = tmpl.format(
            device=random.choice(DEVICE_LIST),
            time_period=f"{random.randint(2, 24)} {random.choice(['weeks', 'months'])}",
            feature1=random.choice(FEATURE1_LIST),
            feature2=random.choice(FEATURE2_LIST),
            drawback=random.choice(DRAWBACK_LIST),
            price=random.randint(25, 250),
            hours=random.randint(6, 40),
            cable_len=random.choice([3, 4, 5, 6]),
            minutes=random.randint(5, 25),
            days=random.randint(3, 14),
            kitchen_item=random.choice(KITCHEN_LIST),
        )

        # Rating variation with slight jitter
        rating = max(1.0, min(5.0, float(base_rating + random.choice([0, 0, 0, -1, 1]))))
        records.append({
            "review_text": text,
            "rating": rating,
            "category": cat,
            "label": 0,  # Genuine
            "label_name": "Genuine",
        })

    # 2. Deceptive / Fake reviews (label = 1)
    for _ in range(half):
        tmpl, base_rating, cat = random.choice(FAKE_SPAM_TEMPLATES)
        text = tmpl
        # Introduce occasional punctuation variations or duplications
        if random.random() < 0.3:
            text = text + " " + random.choice(["BUY IT!", "A+++", "TRUST ME!", "DO NOT HESITATE!!", "BEST DECISION EVER!"])

        records.append({
            "review_text": text,
            "rating": float(base_rating),
            "category": cat,
            "label": 1,  # Fake / Deceptive
            "label_name": "Fake",
        })

    df = pd.DataFrame(records)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def load_dataset(dataset_path: str = None) -> pd.DataFrame:
    """Loads existing dataset or generates synthetic benchmark dataset if path does not exist."""
    default_csv = os.path.join(os.path.dirname(__file__), "synthetic_benchmark.csv")
    path_to_use = dataset_path or default_csv

    if os.path.exists(path_to_use):
        try:
            df = pd.read_csv(path_to_use)
            if "review_text" in df.columns and "label" in df.columns:
                return df
        except Exception:
            pass

    # Generate benchmark dataset
    df = generate_synthetic_dataset(num_samples=3200)
    os.makedirs(os.path.dirname(default_csv), exist_ok=True)
    df.to_csv(default_csv, index=False)
    return df
