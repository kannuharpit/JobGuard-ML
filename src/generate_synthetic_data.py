"""
generate_synthetic_data.py
--------------------------
Generates a realistic synthetic "Fake Job Postings" dataset that mirrors
the structure and statistics of the Kaggle dataset by Shivam Bansal.

Adds deliberate noise/overlap so models produce realistic (non-trivial) metrics:
  - ~17,880 rows, ~4.84% fraudulent
  - ~25% of fraud rows get partially-legit text (hard negatives)
  - ~10% of legit rows get slightly ambiguous phrasing

Run: python src/generate_synthetic_data.py
"""

import os, random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "fake_job_postings.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

N_TOTAL = 17880
N_FRAUD = 866   # ~4.85%
N_LEGIT = N_TOTAL - N_FRAUD

# ── Vocabulary pools ───────────────────────────────────────────────────────────
LEGIT_TITLES = [
    "Software Engineer", "Data Scientist", "Product Manager", "UX Designer",
    "Marketing Manager", "Financial Analyst", "Operations Manager",
    "Business Analyst", "DevOps Engineer", "Senior Java Developer",
    "Machine Learning Engineer", "Full Stack Developer", "Sales Executive",
    "HR Manager", "Project Manager", "Cloud Architect", "QA Engineer",
    "Backend Developer", "Frontend Developer", "Data Analyst",
    "IT Support Specialist", "Network Engineer", "Scrum Master",
    "Content Strategist", "Digital Marketing Specialist", "Account Manager",
    "Research Scientist", "Systems Administrator", "Technical Writer",
    "Supply Chain Analyst", "Healthcare Data Analyst", "Cybersecurity Analyst",
    "Customer Success Manager", "Solutions Architect", "Data Engineer",
    "Growth Hacker", "Brand Manager", "Recruiting Coordinator",
]

AMBIGUOUS_TITLES = [
    "Work From Home Customer Support Agent",
    "Remote Data Entry Specialist",
    "Online Sales Representative",
    "Part-Time Administrative Assistant",
    "Independent Contractor – Flexible Hours",
    "Freelance Social Media Manager",
    "Home-Based Medical Transcriptionist",
    "Virtual Recruiter",
    "Remote Bookkeeper",
    "Online Tutor",
]

FRAUD_TITLES = [
    "Work From Home – Earn $5000 Weekly!", "Online Data Entry – No Experience Needed",
    "Get Paid to Take Surveys!", "Secret Shopper – Immediate Hiring",
    "Envelope Stuffing Job – $20/hour guaranteed", "Make Money Online From Home",
    "Financial Agent Needed Urgently", "Remote Customer Service – No Skills Required",
    "Freelance Jobs – Earn Up To $500 Daily", "Email Processing Agent",
    "Part-Time Data Entry – Earn $800/Week", "Reshipping Coordinator Needed",
    "Social Media Evaluator – Earn From Home", "Virtual Assistant – Earn $40/hr",
    "Be Your Own Boss – Work When You Want", "Earn Cash Fast – No Interview",
    "Mystery Shopper – Weekly Pay", "Home Assembly Work – Great Pay",
]

LEGIT_COMPANIES = [
    "Acme Technologies", "Global Solutions Inc.", "NextGen Consulting",
    "Pinnacle Analytics", "BrightPath Finance", "Synergy Software",
    "Momentum Digital", "Apex Logistics", "Vertex Health Systems",
    "Orion Data Services", "Catalyst Marketing", "Fusion Retail Group",
    "Horizon Energy", "BlueSky Media", "ClearPath Education",
    "Robust Systems", "Alpine Healthcare", "Meridian Consulting",
    "Ironstone Partners", "Vanguard Analytics",
]

INDUSTRIES = [
    "Information Technology and Services", "Marketing and Advertising",
    "Financial Services", "Hospital & Health Care", "Retail",
    "Human Resources", "Computer Software", "Management Consulting",
    "Staffing and Recruiting", "Education Management",
    "Accounting", "Insurance", "Telecommunications",
    "Oil & Energy", "Construction", "Automotive",
    np.nan, np.nan, np.nan,
]

EMP_TYPES   = ["Full-time", "Part-time", "Contract", "Temporary", "Other", np.nan, np.nan]
EXPERIENCE  = ["Not Applicable", "Entry level", "Mid-Senior level", "Associate",
               "Director", "Executive", "Internship", np.nan, np.nan, np.nan]
EDUCATION   = ["Bachelor's Degree", "Master's Degree", "High School or equivalent",
               "Associate Degree", "Some College Coursework Completed",
               "Certification", "Professional", "Doctorate", np.nan, np.nan]
FUNCTIONS   = ["Engineering", "Sales", "Marketing", "Information Technology",
               "Business Development", "Finance", "Human Resources",
               "Product Management", "Design", "Operations", "Education",
               np.nan, np.nan]
CITIES      = ["New York, NY", "San Francisco, CA", "Chicago, IL", "Austin, TX",
               "Seattle, WA", "Boston, MA", "Los Angeles, CA", "Atlanta, GA",
               "Denver, CO", "Miami, FL", "London, GB", "Remote",
               "Toronto, ON", "Sydney, AU", "Berlin, DE"]

# ── Text corpora ───────────────────────────────────────────────────────────────
LEGIT_DESC = [
    "We are looking for an experienced professional to join our growing team.",
    "The ideal candidate will have strong analytical and communication skills.",
    "You will collaborate cross-functionally with product, design, and engineering.",
    "Responsibilities include designing scalable solutions and mentoring junior members.",
    "This role offers an exciting opportunity to make a significant impact.",
    "You will work in an Agile environment with daily standups and sprint planning.",
    "Key duties include developing and maintaining production software.",
    "You will analyze business requirements and translate them into specifications.",
    "The successful candidate will have a proven track record of delivery.",
    "Experience with cloud platforms such as AWS, GCP, or Azure is preferred.",
    "Strong Python or Java skills are required for this position.",
    "You will be responsible for data pipeline development and maintenance.",
    "Our team values innovation, collaboration, and continuous improvement.",
    "We operate at scale, handling billions of requests per month.",
    "You will partner with product managers to define the technical roadmap.",
    "The role requires strong ownership and the ability to drive decisions.",
    "We follow best practices in code review, testing, and CI/CD.",
    "This is a hands-on technical leadership role with a clear growth path.",
]

FRAUD_DESC = [
    "No experience necessary! Anyone can do this job from home.",
    "Earn $500-$2000 per week guaranteed with minimal effort.",
    "Work only 2-3 hours per day and earn a full-time income!",
    "This is a once-in-a-lifetime opportunity to achieve financial freedom.",
    "We are urgently hiring people worldwide with no interview required.",
    "You will receive payment via Western Union or wire transfer weekly.",
    "All you need is a computer and internet connection to get started.",
    "We offer unlimited earning potential with no cap on your income.",
    "Start immediately – no training period, money in your account today.",
    "Join our global network of over 50,000 successful home workers.",
    "Our revolutionary system does 90% of the work automatically!",
    "Limited positions available – apply NOW before it's too late!!!",
    "Make thousands of dollars from the comfort of your own home.",
    "No boss, no commute, no office politics – just money!",
    "Guaranteed income from day one, no selling, no cold calling.",
]

# Ambiguous / overlap sentences used to add noise
NEUTRAL_DESC = [
    "This is a flexible remote-friendly position with great work-life balance.",
    "We welcome candidates from all backgrounds and experience levels.",
    "The position offers competitive compensation based on experience.",
    "Immediate openings available for motivated self-starters.",
    "You will have the opportunity to work with a diverse global team.",
    "This part-time role can be performed from any location.",
    "We are looking for reliable individuals who can manage their own time.",
]

LEGIT_REQS = [
    "Bachelor's degree in Computer Science, Engineering, or related field.",
    "3+ years of professional experience in a similar role.",
    "Proficiency in Python, SQL, and data visualization tools.",
    "Excellent verbal and written communication skills.",
    "Strong problem-solving and critical thinking abilities.",
    "Experience with agile methodologies and version control (Git).",
    "Ability to work independently and as part of a collaborative team.",
    "Knowledge of cloud services (AWS, GCP, or Azure) is a plus.",
    "Experience building and deploying ML models at scale.",
    "MBA or advanced degree preferred for senior positions.",
    "Experience with distributed systems and microservices architecture.",
    "Strong understanding of data structures and algorithms.",
]

FRAUD_REQS = [
    "Must be 18 years or older.",
    "No prior experience or education required.",
    "Must have a valid bank account for direct deposit.",
    "Access to the internet and a computer or smartphone.",
    "Willing to work part-time or full-time hours.",
    "Positive attitude and willingness to learn.",
    "Basic computer skills (email, web browsing).",
    "Must be able to receive packages at your address.",
]

LEGIT_BENEFITS = [
    "Competitive salary and annual performance bonuses.",
    "Comprehensive health, dental, and vision insurance.",
    "401(k) with company matching up to 4%.",
    "Generous paid time off: 20 days PTO + federal holidays.",
    "Flexible remote work options after 90 days.",
    "Employee stock purchase plan (ESPP).",
    "Professional development budget ($2,000/year).",
    "Gym membership reimbursement.",
    "Catered lunches and weekly team events.",
    "Parental leave: 16 weeks fully paid for primary caregivers.",
]

FRAUD_BENEFITS = [
    "Unlimited earning potential!",
    "Work your own hours – no schedule restrictions.",
    "Be your own boss!",
    "No commute – work from the comfort of your home.",
    "Extra cash for vacations and lifestyle upgrades.",
    "Weekly payments directly to your account.",
    "Bonuses paid daily for top performers.",
    "No experience needed – training provided in 10 minutes.",
]


def pick(pool, n):
    return " ".join(random.sample(pool, min(n, len(pool))))


def make_salary(fraudulent):
    if fraudulent:
        return f"{random.randint(50,200)*100}-{random.randint(500,2000)*100}" if random.random() < 0.25 else np.nan
    else:
        base = random.randint(50, 180) * 1000
        return f"{base}-{base + random.randint(10,50)*1000}" if random.random() < 0.38 else np.nan


def make_company_profile(company, fraudulent):
    if fraudulent:
        if random.random() < 0.55:
            return np.nan
        pool = [
            "Global Opportunities Ltd.",
            "We connect talented individuals with flexible income streams.",
            f"Our company has paid out over ${random.randint(1,10)} million to our home workers.",
            "Join thousands of people already working from home successfully.",
        ]
        return pick(pool, random.randint(1, 2))
    else:
        pool = [
            f"{company} is a leading provider of technology solutions with offices worldwide.",
            f"Founded in {random.randint(1990,2018)}, {company} employs over {random.randint(50,5000):,} professionals.",
            "We are committed to innovation, diversity, and delivering value to our clients.",
            f"{company} has been recognized as a Great Place to Work for {random.randint(1,10)} consecutive years.",
            "Our culture emphasizes psychological safety, growth, and technical excellence.",
        ]
        return pick(pool, random.randint(2, 4))


# ── Build rows ─────────────────────────────────────────────────────────────────
rows = []

for i in range(N_LEGIT):
    title   = random.choice(LEGIT_TITLES if random.random() > 0.08 else AMBIGUOUS_TITLES)
    company = random.choice(LEGIT_COMPANIES)

    # 10% of legit rows get some neutral/slightly ambiguous text (hard cases)
    if random.random() < 0.10:
        desc_pool = LEGIT_DESC[:8] + NEUTRAL_DESC
        req_pool  = LEGIT_REQS[:6] + FRAUD_REQS[:2]
    else:
        desc_pool = LEGIT_DESC
        req_pool  = LEGIT_REQS

    rows.append({
        "job_id": i + 1,
        "title": title,
        "location": random.choice(CITIES),
        "department": random.choice(["Engineering","Marketing","Sales","Finance","HR",np.nan]),
        "salary_range": make_salary(False),
        "company_profile": make_company_profile(company, False),
        "description": pick(desc_pool, random.randint(4, 8)),
        "requirements": pick(req_pool, random.randint(3, 6)),
        "benefits": pick(LEGIT_BENEFITS, random.randint(3, 5)) if random.random() < 0.72 else np.nan,
        "telecommuting": int(random.random() < 0.10),
        "has_company_logo": int(random.random() < 0.78),
        "has_questions": int(random.random() < 0.40),
        "employment_type": random.choice(EMP_TYPES),
        "required_experience": random.choice(EXPERIENCE),
        "required_education": random.choice(EDUCATION),
        "industry": random.choice(INDUSTRIES),
        "function": random.choice(FUNCTIONS),
        "fraudulent": 0,
    })

for i in range(N_FRAUD):
    # 25% of fraud rows get some legit-sounding text (hard positives)
    if random.random() < 0.25:
        title     = random.choice(AMBIGUOUS_TITLES)
        desc_pool = FRAUD_DESC[:6] + NEUTRAL_DESC + LEGIT_DESC[:3]
        req_pool  = FRAUD_REQS + LEGIT_REQS[:2]
        ben_pool  = FRAUD_BENEFITS[:4] + LEGIT_BENEFITS[:2]
        has_logo  = int(random.random() < 0.35)
        has_q     = int(random.random() < 0.25)
    else:
        title     = random.choice(FRAUD_TITLES)
        desc_pool = FRAUD_DESC
        req_pool  = FRAUD_REQS
        ben_pool  = FRAUD_BENEFITS
        has_logo  = int(random.random() < 0.12)
        has_q     = int(random.random() < 0.08)

    rows.append({
        "job_id": N_LEGIT + i + 1,
        "title": title,
        "location": random.choice(CITIES + [np.nan]),
        "department": np.nan if random.random() < 0.72 else random.choice(["Sales","Admin"]),
        "salary_range": make_salary(True),
        "company_profile": make_company_profile(None, True),
        "description": pick(desc_pool, random.randint(3, 6)),
        "requirements": pick(req_pool, random.randint(2, 4)),
        "benefits": pick(ben_pool, random.randint(2, 4)),
        "telecommuting": int(random.random() < 0.75),
        "has_company_logo": has_logo,
        "has_questions": has_q,
        "employment_type": random.choice(["Part-time","Other",np.nan,np.nan]),
        "required_experience": random.choice(["Not Applicable",np.nan,np.nan]),
        "required_education": random.choice([np.nan,np.nan,"High School or equivalent"]),
        "industry": np.nan if random.random() < 0.62 else random.choice(INDUSTRIES),
        "function": np.nan if random.random() < 0.68 else random.choice(FUNCTIONS),
        "fraudulent": 1,
    })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(OUT_PATH, index=False)
print(f"[OK] Dataset saved: {OUT_PATH}")
print(f"     Rows: {len(df)} | Fraud: {df['fraudulent'].sum()} ({df['fraudulent'].mean():.2%})")
