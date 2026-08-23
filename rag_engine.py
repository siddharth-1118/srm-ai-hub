"""
Self-contained retrieval engine for the SRM Admissions AI Assistant.

No third-party AI models: retrieval is powered by a TF-IDF vectorizer
(scikit-learn) trained on the brochure text, combined with a lightweight
keyword (BM25-lite) score. The trained index is saved to a local pickle so
a deployed web server only loads precomputed data - no training at runtime.

Multi-document support: every brochure/poster is an entry in DOCUMENTS.
To add a new poster, drop the PDF into the project folder, add one entry
to DOCUMENTS (optionally with curated facts), and run `python rag_engine.py`
to rebuild the index. All paths are relative to this file, so the app runs
from any folder or server.
"""

import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional OCR support for image-based PDFs.  When pypdf extracts no text
# from a page, we render it to an image and run RapidOCR (ONNX-based,
# no external binaries required).  If the libraries are not installed the
# engine falls back silently to skipping the page.
try:
    import pymupdf as _fitz  # type: ignore[import-untyped]
    from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]
    _HAS_OCR = True
    _OCR_ENGINE = None  # lazy-loaded on first use
except ImportError:
    _HAS_OCR = False
    _OCR_ENGINE = None

_ocr_log = logging.getLogger(__name__)


def _get_ocr_engine():
    """Lazy-load the RapidOCR engine only when first needed."""
    global _OCR_ENGINE  # noqa: PLW0603
    if _OCR_ENGINE is None:
        if _HAS_OCR:
            _ocr_log.info("Initializing OCR engine (first use)...")
            _OCR_ENGINE = RapidOCR()
            _ocr_log.info("OCR enabled: image-based PDFs will be text-extracted via RapidOCR.")
        else:
            _ocr_log.info("OCR disabled: install pymupdf + rapidocr-onnxruntime for image PDF support.")
            return None
    return _OCR_ENGINE

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "srm-admission-brochure-2026-2027.pdf"
INDEX_PATH = BASE_DIR / "vector_index.pkl"

# Common query function words that must not drive keyword ranking
# (e.g. "all pg programs in srm..." must not rank footer chunks on "srm institute").
STOPWORDS = {
    "a", "about", "all", "also", "an", "and", "any", "are", "as", "at", "be", "by",
    "can", "could", "do", "does", "for", "from", "get", "give", "have", "how", "i",
    "in", "into", "is", "it", "its", "list", "me", "no", "not", "of", "on", "or",
    "our", "please", "show", "tell", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "to", "want", "was", "we", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your"
}

# Domain synonym expansion: bridges the gap between how students phrase a question
# and how the brochure spells things ("pg" -> "postgraduate", "btech" -> "b.tech").
QUERY_SYNONYMS = {
    "pg": "pg postgraduate",
    "postgrad": "postgraduate",
    "postgraduate": "postgraduate",
    "ug": "ug undergraduate",
    "undergraduate": "undergraduate",
    "btech": "btech",
    "mtech": "mtech",
    "tech": "tech btech mtech",
    "msc": "msc",
    "bsc": "bsc",
    "mcom": "mcom",
    "bcom": "bcom",
    "mba": "mba",
    "bba": "bba",
    "mca": "mca",
    "mph": "mph",
    "phd": "phd",
    "llm": "llm",
    "srmist": "srmist srm",
    "entrance": "entrance exam",
    "exam": "exam entrance",
    "srmjee": "srmjee srmjeee srmjeeh srmjeel srmjeem",
    "srmjeee": "srmjeee srmjee",
    "srmjeeh": "srmjeeh srmjee health science",
    "srmjeel": "srmjeel srmjee law",
    "srmjeem": "srmjeem srmjee management",
    "pgeta": "pgeta architecture",
}

# Curated Postgraduate (PG) program directory extracted from the 2026-27 brochure
# (pages 4-7). Injected as chunks so "PG / postgraduate / M.Sc / MBA..." queries
# always retrieve the real program lists instead of footer or contact-page noise.
PG_PROGRAMS = [
    {
        "school": "Architecture and Interior Design",
        "page": 4,
        "programs": [
            "M.Arch - Architectural Design (Entrance Exam: PGETA)",
            "M.Des - Public Space Design",
        ],
    },
    {
        "school": "Agricultural Sciences (Baburayanpettai Campus)",
        "page": 4,
        "programs": [
            "M.Sc. (Agriculture) - Agronomy | Agricultural Economics | Agricultural Extension Education | Entomology | Genetics and Plant Breeding | Plant Pathology | Soil Science",
            "M.Sc. (Horticulture) - Floriculture and Landscaping | Fruit Science | Vegetable Science",
            "M.Sc. Yoga (fully residential)",
        ],
    },
    {
        "school": "Science and Humanities",
        "page": 4,
        "programs": [
            "M.Com - Commerce | Accounting & Finance",
            "M.A. - Journalism and Mass Communication | English",
            "M.S - Computer Science W/S Data Science | Fintech | Computer Science W/S Agentic Artificial Intelligence and Machine Learning",
            "M.Sc. - Applied Data Science | Organic Chemistry | Biochemistry | Biotechnology | Chemistry | Computer Science | Information Technology | Mathematics | Physics | Visual Communication | Yoga | Disaster Management | Computer Science W/S Full Stack Development | Computer Science W/S Cyber Security | Computer Science W/S AIML | Counselling Psychology | Fashion Designing | Applied Physics W/S Semiconductor Technology | Applied Physics W/S Quantum Technology | Mathematics and Computing",
            "MCA - Computer Applications | Computer Applications W/S Generative AI",
            "M.S.W - Social Work (Human Resource Management | Medical & Psychiatry and Community Development)",
            "PG Diploma - Journalism and Mass Communication",
        ],
    },
    {
        "school": "Law",
        "page": 5,
        "programs": [
            "LL.M - Criminal Law and Criminal Justice | International Trade Law & IPR | Corporate Law | Cyber Law and Security",
            "MPP - Master of Public Policy",
        ],
    },
    {
        "school": "Hotel and Catering Management",
        "page": 5,
        "programs": [
            "PG Diploma - Culinary Arts",
        ],
    },
    {
        "school": "Management",
        "page": 5,
        "programs": [
            "MBA - Business Administration | Banking and Financial Services | Health Care and Hospital Management | Artificial Intelligence and Data Science | Business Analytics | Digital Marketing | Financial Services along with NSE | Sports Management | Logistics and Supply Chain Management | Working Professionals Programs",
            "MMS - Master of Management Studies",
            "PGDM-X - Post Graduate Diploma in Management for Executives",
        ],
    },
    {
        "school": "Medicine (Health Sciences)",
        "page": 6,
        "programs": [
            "MD - Anatomy | Physiology | Biochemistry | Pharmacology | Pathology | Microbiology | Community Medicine | General Medicine | Paediatrics | Psychiatry | Dermatology, Venereology and Leprosy | Respiratory Medicine | Radio-Diagnosis | Anaesthesiology | Emergency Medicine | Forensic Medicine & Toxicology | Immuno Haematology & Blood Transfusion",
            "MS - Obstetrics & Gynaecology | General Surgery | Orthopaedics | Ophthalmology | Otorhinolaryngology (ENT)",
            "DM - Cardiology | Nephrology | Neurology | Critical Care Medicine | Medical Gastroenterology",
            "M.Ch - Cardio Vascular & Thoracic Surgery | Neurosurgery | Plastic & Reconstructive Surgery | Pediatric Surgery | Urology",
            "Masters (Allied Health) - Advanced Care Paramedics | Anaesthesia and Operation Theatre Technology | Dialysis Therapy | Medical Laboratory Science | Medical Radiology and Imaging Technology | Nutrition and Dietetics | Respiratory Technology | Optometry",
            "Masters (Allied Health) - Medical Anatomy | Medical Physiology | Medical Biochemistry | Medical Microbiology | Cardiovascular Sciences (Echocardiography) | Cardio Perfusion Technology | Critical Care Technology | Clinical Research | Neuroscience Technology | Sports and Exercise Psychology | Urology Technology",
            "MHA - Hospital Administration",
            "M.Sc - Audiology | Speech Language Pathology",
            "M.Clin. Psy - Master in Clinical Psychology",
            "Fellowship - Emergency Medicine | Reproductive Medicine",
        ],
    },
    {
        "school": "Dentistry",
        "page": 6,
        "programs": [
            "MDS - Orthodontics & Dento Facial Orthopedics | Prosthodontics and Crown & Bridge | Conservative Dentistry and Endodontics | Oral and Maxillofacial Surgery | Periodontology | Oral Pathology & Microbiology | Oral Medicine | Paediatric and Preventive Dentistry | Public Health Dentistry | Periodontics",
            "Psy.D - Clinical Psychology",
            "PG Diploma - Clinical Embryology",
        ],
    },
    {
        "school": "Pharmacy",
        "page": 7,
        "programs": [
            "M.Pharm - Pharmaceutics | Pharmaceutical Regulatory Affairs | Pharmaceutical Quality Assurance | Pharmaceutical Analysis | Pharmacology | Pharmacognosy | Pharmaceutical Chemistry | Pharmacy Practice",
        ],
    },
    {
        "school": "Occupational Therapy",
        "page": 7,
        "programs": [
            "M.O.T - Musculoskeletal Sciences | Paediatrics & Neonatology | Neurosciences | Mental Health | Hand Rehabilitation",
        ],
    },
    {
        "school": "Physiotherapy",
        "page": 7,
        "programs": [
            "M.P.T - Musculoskeletal Science | Neuroscience | Cardio-Pulmonary Science | Sports Science | Pediatrics and Neonatal Sciences | Obstetrics and Gynaecology Science | Community Rehabilitation",
        ],
    },
    {
        "school": "Nursing",
        "page": 7,
        "programs": [
            "M.Sc - Medical Surgical Nursing | Obstetrics and Gynaecology Nursing | Paediatric Nursing | Psychiatric Nursing | Community Health Nursing | Nurse Practitioner in Critical Care",
            "Post Basic Diploma - Critical Care Nursing | Emergency and Disaster Nursing | Operation Room Nursing",
        ],
    },
    {
        "school": "Public Health",
        "page": 7,
        "programs": [
            "MPH - Public Health | Applied Health Research",
            "M.Sc - Biostatistics and Epidemiology | Health Data Science",
            "MPH (Integrated) - Public Health",
        ],
    },
]

# Lightweight curated overview of undergraduate programs (from the same brochure)
# so common UG questions ("B.Tech", "B.Sc", "BBA"... ) also retrieve accurate lists.
UG_OVERVIEW = (
    "SRMIST Undergraduate (UG) Programs Overview: Undergraduate degrees include "
    "B.Tech (Engineering and Technology, entrance exam SRMJEEE) with majors and "
    "specialisations across Aeronautical, Aerospace, AI, Biomedical, Biotechnology, "
    "Chemical, Civil, Computer Science, ECE, EEE, IT, Mechanical, Mechatronics, "
    "Nanotechnology and more, plus Integrated M.Tech. Also offered: B.Arch (NATA), "
    "B.Des, B.Sc (Hons), B.Com, B.A., B.S., BCA, BBA, B.Ed, B.Pharm, Pharm.D, BDS (NEET), "
    "MBBS (NEET), B.O.T, B.P.T, B.Sc Nursing, B.ASLP, LL.B (5-year, SRMJEEL), "
    "BBA-LL.B, B.Com-LL.B and BA-LL.B, B.Sc Hotel and Hospitality Administration, "
    "and Diplomas."
)

# Curated B.Tech majors and specialisations (from page 3 of the brochure).
UG_ENGINEERING = [
    "B.Tech majors (SRMJEEE): Aeronautical | Aerospace | Artificial Intelligence | Artificial Engineering & Future Technologies | Automation and Robotics | Automobile | Biomedical | Biotechnology | Biotechnology (Computational Biology) | Biotechnology (Food Technology) | Biomedical (Machine Intelligence) | Chemical | Civil | Civil Engineering with Computer Application | Computer Science | Computer Science and Business System | Computer Science (Data Science) | Computer Science (DevSecOps) | Electrical and Electronics | Electronics and Communication | Electronics and Computer | Electronics and Instrumentation | Electronics Engineering (VLSI Design and Technology) | Information Technology | Mathematics and Computing | Mechanical | Mechanical (Automation & Robotics) | Mechatronics | Nanotechnology | Software Product Engineering",
    "B.Tech with Specialisation (W/S): Automotive Electronics (Automobile); Genetic Engineering | Regenerative Medicine (Biotechnology); Artificial Intelligence and Machine Learning | Big Data Analytics | Block Chain Technology | Cloud Computing | Cyber Security | Computer Networking | Gaming Technology | Information Technology | Internet of Things (IOT) (Computer Science)",
    "Integrated M.Tech (5 year, IDEAL - Interdisciplinary Experiential Active Learning, dual specialisation)",
]

# Curated FAQ entries: honest answers for questions the brochure does NOT cover
# (so the assistant never fabricates or answers with unrelated sections).
CURATED_FAQ = [
    "Tuition Fees: The SRM Admission Brochure 2026-27 does not publish tuition fee amounts. "
    "The only fee mention is that Sports Quota students are exempted from payment of book fees and "
    "examination fees. For the exact tuition fee structure, refer to the SRM website: www.srmist.edu.in",
    "Hostel and Accommodation: SRMIST provides hostel accommodation at the Kattankulathur Campus for all "
    "batches. Hostel fees, room types and booking schedules are available in the hostel circulars. "
    "Contact the admissions helpline (080 - 6908 7000) or visit https://sp.srmist.edu.in/srmiststudentportal for details.",
    "Application Deadline and Important Dates: The brochure does not state exact application deadlines. "
    "It lists entrance examination phases (SRMJEEE B.Tech 2026 Phase I, II, III; SRMJEEH Health Science "
    "UG / PG 2026; SRMJEEL Law 2026; SRMJEEM) but not calendar dates. "
    "Check the SRM website www.srmist.edu.in for the exact application schedule and last dates.",
    "Admission Process: Admissions are through SRM's own entrance examinations - SRMJEEE (B.Tech), "
    "SRMJEEH HEALTH SCIENCE (UG / PG / POST PG), SRMJEEL (Law), SRMJEEM (Management) and PGETA (M.Arch) - "
    "plus NATA or NEET for specific programs. Apply online at https://applications.srmist.edu.in, "
    "or call the Admissions Helpline: 080 - 6908 7000.",
    "Scholarships: SRMIST offers the Founder's Scholarship, SRM Merit Scholarship, Socio-Economic "
    "Scholarship, Differently-Abled Scholarship, Defence Scholarship, SRM Arts and Culture Scholarship, "
    "Employee Ward Scholarship, Alumni Scholarship, Academic Excellence Scholarship and Unnat Bharat "
    "Abhiyan Scholarship. Annually the scholarship budgeted outlay exceeds 50 Crores, benefitting "
    "3000+ students.",
    "Campuses: SRMIST has campuses in Tamil Nadu, Andhra Pradesh, Uttar Pradesh, Haryana and Sikkim. "
    "Agricultural Sciences programs, and the fully residential B.Sc Physical Education, M.Sc Yoga and "
    "Diploma in Yoga, are offered at the Baburayanpettai Campus. Vadapalani Campus hosts Sports Quota "
    "students such as Olympians Ms. Nethra Kumanan and Mr. Prithviraj Thondaiman.",
    "Eligibility Criteria: Eligibility depends on the program and is based on entrance exam performance "
    "(SRMJEEE / SRMJEEH / SRMJEEL / SRMJEEM / NATA / NEET) and 12th standard marks - for health sciences, "
    "Physics, Chemistry and Biology (PCB). Program nomenclatures, duration and eligibility are subject "
    "to change as per regulations; refer to srmist.edu.in/admission-india for exact criteria.",
    "Sports Quota Admissions: Sports quota admissions are offered to athletes who have represented the "
    "state or nation. Sports Quota students receive free boarding and lodging and are exempted from "
    "book fees and examination fees. The Directorate of Sports advertises in major newspapers across "
    "India, and notable Olympians like Ms. Nethra Kumanan and Mr. Prithviraj Thondaiman were admitted "
    "under this scheme at the Vadapalani Campus.",
    "Admission Cutoffs: The SRM Admission Brochure 2026-27 does not publish cutoff marks or minimum "
    "score requirements. Admissions are decided by performance in SRM's entrance examinations "
    "(SRMJEEE for B.Tech, SRMJEEH HEALTH SCIENCE for UG/PG health science programs, SRMJEEL for Law, "
    "SRMJEEM for Management, plus NATA or NEET for specific programs) combined with 12th standard "
    "marks - for health sciences, Physics, Chemistry and Biology (PCB). Cutoffs vary by program, "
    "campus and year, so check https://www.srmist.edu.in/admission-india for the latest closing ranks "
    "and eligibility.",
    "Rankings and Accreditation: The brochure's accolades include SRMIST ranked 25th in India in the "
    "Nature Index Ranking (Sep 2022 - Aug 2023), the FICCI University of the Year Award (11-30 years "
    "in existence category), the AICTE Swachhata Ranking Award for Clean and Smart Campus, CII "
    "Industrial Intellectual Property Awards 2021, the STEM impact award, the Green Metric Award and "
    "the AICTE-CII Ind Pact Award. At the Kattankulathur Campus, B.Tech programs in Civil, Mechanical, "
    "EEE, ECE, IT, CSE, Automobile, E&I and Software Engineering are accredited by ABET, and Biotech, "
    "Mechanical, Civil, EEE and ECE are NBA accredited. The brochure does not mention NAAC grading.",
    "Campus Infrastructure: SRMIST advertises world-class faculty and infrastructure. The brochure "
    "highlights state-of-the-art research facilities (Rs.125 Crores) and more than 500 research "
    "projects at an outlay of Rs.230 Crores. Sports infrastructure and facilities include "
    "world-class floodlit playfields, a lush green cricket ground with a 10,000-seat gallery, a "
    "400-meter standard athletic track, and an International Standard Swimming Pool complex with "
    "competition (50m x 25m x 2m), training (25m x 15m x 1.5m) and diving (25m x 20m x 5m) pools. "
    "Athletes receive nutritious meals, medical care, physiotherapy, injury treatment and an "
    "insurance cover of Rs.2,00,000, and SRMIST has produced 17 Chess Grand Masters so far.",
    "Global Exposure: Through the Semester Abroad Program (SAP), SRMIST students can spend a semester "
    "at one of 150+ reputed partner universities worldwide, with SAP scholarships available. The "
    "brochure also highlights the IDEAL (Interdisciplinary Experiential Active Learning) approach for "
    "the integrated M.Tech, offering dual specialisation and international exposure.",
]

# Curated hostel fee summaries — structured data from hostel circular PDFs so that
# "hostel fees" queries always retrieve the real fee tables instead of the brochure FAQ.
HOSTEL_FEES_FIRSTYEAR_BOYS = [
    "First Year B.Arch/B.Des Boys Hostel Fees 2026-27 (Kattankulathur Campus): "
    "AC with attached washroom 2-sharing N Block: Hostel Rs 2,00,000 + Mess Rs 73,000 = Total Rs 2,73,000. "
    "AC with attached washroom 3-sharing Green Pearl (off-campus): Hostel Rs 1,45,000 + Mess Rs 1,25,000 = Total Rs 2,70,000. "
    "AC with attached washroom 3-sharing N Block: Hostel Rs 1,80,000 + Mess Rs 73,000 = Total Rs 2,53,000. "
    "AC with attached washroom 2-sharing Adhiyaman: Hostel Rs 1,70,000 + Mess Rs 73,000 = Total Rs 2,43,000. "
    "AC with attached washroom 3-sharing Adhiyaman: Hostel Rs 1,60,000 + Mess Rs 73,000 = Total Rs 2,33,000. "
    "AC with attached washroom 4-sharing Adhiyaman: Hostel Rs 1,50,000 + Mess Rs 73,000 = Total Rs 2,23,000. "
    "Non-AC with attached washroom 3-sharing Green Pearl (off-campus): Hostel Rs 65,000 + Mess Rs 1,25,000 = Total Rs 1,90,000. "
    "AC with common washroom 4-5 sharing Oori: Hostel Rs 1,15,000 + Mess Rs 73,000 = Total Rs 1,88,000. "
    "Non-AC with attached washroom 2-sharing N Block: Hostel Rs 1,00,000 + Mess Rs 73,000 = Total Rs 1,73,000. "
    "Non-AC with attached washroom 3-sharing N Block: Hostel Rs 90,000 + Mess Rs 73,000 = Total Rs 1,63,000. "
    "Non-AC with attached washroom 3-sharing Adhiyaman: Hostel Rs 80,000 + Mess Rs 73,000 = Total Rs 1,53,000. "
    "Non-AC with common washroom 4-5 sharing Kaari: Hostel Rs 45,000 + Mess Rs 73,000 = Total Rs 1,18,000. "
    "Non-AC with attached washroom 7-sharing JA Block 2 (off-campus): Hostel Rs 37,000 + Mess Rs 73,000 = Total Rs 1,10,000. "
    "Boys laundry fees: Rs 7,500 (optional). Booking starts 08-July-2026 at 10:00 AM. "
    "Book at https://sp.srmist.edu.in/srmiststudentportal",
    "First Year B.Arch/B.Des Girls Hostel Fees 2026-27 (Kattankulathur Campus): "
    "AC with attached washroom 2-sharing Kalpana Chawla: Hostel Rs 1,97,000 + Mess Rs 73,000 = Total Rs 2,70,000. "
    "AC with attached washroom 4-sharing Kalpana Chawla: Hostel Rs 1,50,000 + Mess Rs 73,000 = Total Rs 2,23,000. "
    "AC with attached washroom 4-sharing (bunker cot) Meenakshi: Hostel Rs 1,40,000 + Mess Rs 73,000 = Total Rs 2,13,000. "
    "Non-AC with attached washroom 3-sharing Meenakshi: Hostel Rs 80,000 + Mess Rs 73,000 = Total Rs 1,53,000. "
    "Non-AC with common washroom 2-sharing Thamarai: Hostel Rs 60,000 + Mess Rs 73,000 = Total Rs 1,33,000. "
    "Non-AC with common washroom 3-sharing Mullai: Hostel Rs 50,000 + Mess Rs 73,000 = Total Rs 1,23,000. "
    "Non-AC with common washroom 6-sharing Senbagam: Hostel Rs 45,000 + Mess Rs 73,000 = Total Rs 1,18,000. "
    "Girls laundry fees: Rs 8,500 (optional). Booking starts 08-July-2026 at 10:00 AM. "
    "Book at https://sp.srmist.edu.in/srmiststudentportal",
]

HOSTEL_FEES_SENIOR_BOYS_2025 = [
    "Senior Boys Hostel Fees 2025-26 (FSH, LAW, MGT — Kattankulathur Campus): "
    "Booking opens 16-April-2025 at 4:00 PM. "
    "AC with common washroom 4-sharing Pierre Fauchard (PF): Hostel Rs 1,15,000 + Mess Rs 73,000 = Total Rs 1,88,000. "
    "Non-AC with common washroom 3-sharing Pierre Fauchard (PF): Hostel Rs 50,000 + Mess Rs 73,000 = Total Rs 1,23,000. "
    "Non-AC with common washroom 4-sharing Pierre Fauchard (PF): Hostel Rs 45,000 + Mess Rs 73,000 = Total Rs 1,18,000. "
    "Laundry fees: Rs 7,500 (optional). Hostel fees once paid will not be refunded. "
    "Book at https://sp.srmist.edu.in/srmiststudentportal",
]

HOSTEL_FEES_3RD_YEAR_GIRLS_2025 = [
    "Third Year Girls Hostel Fees 2025-26 — B.Tech/M.Tech/B.Arch/B.Des (Kattankulathur Campus): "
    "Booking opens 10-February-2025 at 5:00 PM. "
    "AC with attached washroom 2-sharing Kopperundevi (M Block): Hostel Rs 1,97,000 + Mess Rs 73,000 = Total Rs 2,70,000. "
    "Non-AC with attached washroom 2-sharing Kopperundevi: Hostel Rs 92,000 + Mess Rs 73,000 = Total Rs 1,65,000. "
    "Non-AC with attached washroom 2-sharing ESQ-B: Hostel Rs 85,000 + Mess Rs 73,000 = Total Rs 1,58,000. "
    "Non-AC with attached washroom 3-sharing ESQ-B: Hostel Rs 75,000 + Mess Rs 73,000 = Total Rs 1,48,000. "
    "Non-AC with common washroom 3-sharing Senbagam: Hostel Rs 50,000 + Mess Rs 73,000 = Total Rs 1,23,000. "
    "Non-AC with common washroom 6-sharing Senbagam: Hostel Rs 45,000 + Mess Rs 73,000 = Total Rs 1,18,000. "
    "Laundry fees: Rs 8,500 (optional). "
    "Book at https://sp.srmist.edu.in/srmiststudentportal",
]

HOSTEL_FEES_4TH_5TH_YEAR_GIRLS_2026 = [
    "Fourth/Fifth Year Girls Hostel Fees 2026-27 — B.Tech/B.Arch/B.Des/M.Tech (Kattankulathur Campus): "
    "Booking opens 13-February-2026 at 5:00 PM. "
    "AC with attached washroom 3-sharing Sannasi C: Hostel Rs 1,60,000 + Mess Rs 73,000 = Total Rs 2,33,000. "
    "Non-AC with attached washroom 3-sharing Sannasi C: Hostel Rs 80,000 + Mess Rs 73,000 = Total Rs 1,53,000. "
    "Non-AC with common washroom 2-sharing Malligai: Hostel Rs 60,000 + Mess Rs 73,000 = Total Rs 1,33,000. "
    "Non-AC with common washroom 3-sharing Senbagam: Hostel Rs 50,000 + Mess Rs 73,000 = Total Rs 1,23,000. "
    "Non-AC with common washroom 6-sharing Senbagam: Hostel Rs 45,000 + Mess Rs 73,000 = Total Rs 1,18,000. "
    "Girls laundry fees: Rs 8,500 (optional). "
    "Book at https://sp.srmist.edu.in/srmiststudentportal",
]

# Registry of documents in the knowledge base. Add new brochures/posters here.
DOCUMENTS = [
    {
        "id": "srm-admissions-2026-27",
        "file": PDF_PATH,
        "title": "SRM Admission Brochure 2026-27",
        "curated": PG_PROGRAMS,
        "ug_overview": UG_OVERVIEW,
        "ug_engineering": UG_ENGINEERING,
        "faq": CURATED_FAQ,
    },
    {
        "id": "hostel-firstyear-boys-arch-des-2026",
        "file": BASE_DIR / "circular-for-all-first-year-b-arch-bdes-2026-boys.pdf",
        "title": "Hostel Fees 2026-27 — First Year B.Arch/B.Des Boys",
        "hostel_curated": HOSTEL_FEES_FIRSTYEAR_BOYS[:1],  # boys portion
    },
    {
        "id": "hostel-firstyear-girls-arch-des-2026",
        "file": BASE_DIR / "circular-for-all-first-year-b-arch-bdes-2026-girls.pdf",
        "title": "Hostel Fees 2026-27 — First Year B.Arch/B.Des Girls",
        "hostel_curated": HOSTEL_FEES_FIRSTYEAR_BOYS[1:],  # girls portion
    },
    {
        "id": "hostel-firstyear-boys-booking",
        "file": BASE_DIR / "hostel-booking-first-year-boys.pdf",
        "title": "Hostel Booking — First Year Boys",
    },
    {
        "id": "hostel-senior-boys-fsh-law-mgt-2025",
        "file": BASE_DIR / "hostel-circular-fsh-law-mgt-all-senior-boys.pdf",
        "title": "Hostel Booking Schedule 2025-26 — Senior Boys (FSH/LAW/MGT)",
        "hostel_curated": HOSTEL_FEES_SENIOR_BOYS_2025,
    },
    {
        "id": "hostel-girls-3rd-year-btech-mtech-2025",
        "file": BASE_DIR / "srm-hostels-booking-schedule-girls-3rd-year-btech-mtech.pdf",
        "title": "Hostel Booking Schedule 2025-26 — 3rd Year Girls B.Tech/M.Tech",
        "hostel_curated": HOSTEL_FEES_3RD_YEAR_GIRLS_2025,
    },
    {
        "id": "hostel-girls-4th-5th-year-2026",
        "file": BASE_DIR / "srm-hostels-booking-schedule-girls-4th-5th-year-btech-mtech.pdf",
        "title": "Hostel Fees 2026-27 — 4th/5th Year Girls B.Tech/B.Arch/B.Des/M.Tech",
        "hostel_curated": HOSTEL_FEES_4TH_5TH_YEAR_GIRLS_2026,
    },
    {
        "id": "hostel-senior-students-2023-24",
        "file": BASE_DIR / "srm-senior-students-hostel-booking-schedule-2023-24.pdf",
        "title": "Hostel Booking Schedule 2023-24 — Senior Students",
    },
]


class RAGEngine:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else DOCUMENTS
        self.index_path = INDEX_PATH
        self.chunks = []
        self.vectorizer = None
        self.tfidf = None
        self.tfidf_norms = None

    # ------------------------------------------------------------------ build

    @staticmethod
    def _clean_text(text):
        """Normalize raw PDF text: fix ligatures, squished tokens, whitespace."""
        text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = " ".join(lines)
        # Fix PDF text extraction squishing (e.g. Salary65LPA -> Salary 65 LPA)
        cleaned = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', cleaned)
        cleaned = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', cleaned)
        cleaned = re.sub(r'(\))([0-9])', r'\1 \2', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned

    @staticmethod
    def _split_chunks(cleaned_text, chunk_size=350, overlap=80):
        """Sentence/word-boundary aware chunking of a cleaned page."""
        start = 0
        while start < len(cleaned_text):
            end = start + chunk_size
            if end < len(cleaned_text):
                period_idx = cleaned_text.rfind('. ', start + chunk_size // 2, end)
                if period_idx > start + chunk_size // 3:
                    end = period_idx + 1
                else:
                    space_idx = cleaned_text.rfind(' ', start + chunk_size // 2, end)
                    if space_idx > start + chunk_size // 3:
                        end = space_idx
            chunk_text = cleaned_text[start:end].strip()
            if len(chunk_text) > 30:  # ignore tiny useless fragments
                yield chunk_text
            start = end - overlap
            if start >= len(cleaned_text):
                break

    def _ocr_page(self, path, page_idx, dpi=200):
        """Render a PDF page to an image and run OCR to extract text."""
        ocr = _get_ocr_engine()
        if ocr is None:
            return ""
        try:
            doc = _fitz.open(str(path))
            if page_idx >= len(doc):
                doc.close()
                return ""
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            doc.close()
            result, _ = ocr(img_bytes)
            if result:
                return " ".join(item[1] for item in result)
        except Exception as e:
            _ocr_log.warning("OCR failed for %s page %d: %s", path, page_idx + 1, e)
        return ""

    def _extract_document(self, doc):
        """Extract text from one PDF and return (reader, page_chunks).

        Falls back to OCR for image-based pages where pypdf extracts no text.
        """
        path = doc["file"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")
        reader = pypdf.PdfReader(str(path))
        chunks = []
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            # If pypdf returned nothing, try OCR for image-based pages
            if not text or not text.strip():
                text = self._ocr_page(path, page_idx)
            if not text or not text.strip():
                continue
            for chunk_text in self._split_chunks(self._clean_text(text)):
                chunks.append({
                    "doc_id": doc["id"],
                    "doc_title": doc["title"],
                    "page": page_idx + 1,
                    "text": chunk_text,
                })
        return reader, chunks

    def _chunk(self, doc, text, page, curated=False, kind=None, school=None):
        return {
            "doc_id": doc["id"],
            "doc_title": doc["title"],
            "page": page,
            "text": text,
            "curated": curated,
            "kind": kind,
            "school": school,
        }

    def _inject_srm_placements(self, doc, chunks, reader):
        """Brochure-specific: inject a human-readable placements summary chunk."""
        full_text = ""
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
            full_text += f"\n[PAGE {page_idx + 1}]\n" + text

        placement_match = re.search(
            r'([\d,]+)\+?\s*\n?Job Offers.*?Highest Salary\s*([\d]+)\s*LPA.*?Global Offers\s*([\d]+)\+.*?'
            r'20 LPA.*?([\d,]+)\+.*?Companies Visited\s*([\d,]+)\+.*?Global Capability.*?([\d]+)\+',
            full_text, re.DOTALL | re.IGNORECASE
        )
        if placement_match:
                g = placement_match
                synthetic = (
                    f"SRMIST Placement Statistics 2025: "
                    f"Total Job Offers received by students: {g.group(1)}+. "
                    f"Highest salary package offered: {g.group(2)} LPA. "
                    f"International / Global Job Offers: {g.group(3)}+. "
                    f"Jobs with package above 20 LPA: {g.group(4)}+. "
                    f"Total companies that visited campus for recruitment: {g.group(5)}+. "
                    f"Global Capability Centres (GCC) that hired: {g.group(6)}+. "
                    f"Top recruiters include Amazon, Google, PayPal, Morgan Stanley, Deloitte, JP Morgan, TCS, Wipro, Cognizant, Capgemini."
                )
                chunks.insert(0, self._chunk(doc, synthetic, 9, curated=True, kind="placements"))
        else:
            if 'PLACEMENTS 2025' in full_text and 'LPA' in full_text:
                page9_match = re.search(r'\[PAGE 9\](.*?)(?:\[PAGE 10\]|$)', full_text, re.DOTALL)
                if page9_match:
                    raw = re.sub(r'\s+', ' ', page9_match.group(1)).strip()
                    chunks.insert(0, self._chunk(
                        doc, "SRMIST Placement and Recruitment Data 2025: " + raw[:600], 9,
                        curated=True, kind="placements"))


    def _inject_curated_chunks(self, doc, chunks, reader):
        """Add curated program chunks so program queries always retrieve accurate lists."""
        ug_overview = doc.get("ug_overview")
        if ug_overview:
            chunks.insert(0, self._chunk(doc, ug_overview, 3, curated=True, kind="ug_overview"))

        ug_engineering = doc.get("ug_engineering") or []
        for text in reversed(ug_engineering):
            chunks.insert(0, self._chunk(doc, text, 3, curated=True, kind="ug_engineering"))

        for text in (doc.get("faq") or []):
            chunks.append(self._chunk(doc, text, 4, curated=True, kind="faq"))

        curated = doc.get("curated") or []
        if curated:
            schools = ", ".join(e["school"] for e in curated)
            overview = (
                f"{doc['title']} Postgraduate (PG) Programs Overview: "
                f"PG degrees are offered across these schools: {schools}. "
                "Postgraduate programs include M.Arch, M.Des, M.Sc, M.Com, M.A., M.S, MCA, M.S.W, LL.M, "
                "MPP, MBA, MMS, PGDM-X, MD, MS, DM, M.Ch, MDS, Psy.D, MHA, M.Pharm, M.O.T, M.P.T, "
                "M.Sc Nursing and MPH. An Integrated M.Tech (5 year) is offered under Engineering. "
                "Admission to most health science PG programs is through SRMJEEH HEALTH SCIENCE - PG, "
                "M.Arch through PGETA, Management programs through SRMJEEM, and Medicine/Dentistry "
                "through NEET counselling."
            )
            chunks.insert(0, self._chunk(doc, overview, 4, curated=True, kind="pg_overview"))
            for entry in curated:
                text = f"{entry['school']} - Postgraduate Programs (PG): " + " | ".join(entry["programs"])
                chunks.append(self._chunk(
                    doc, text, entry["page"], curated=True, kind="pg_school",
                    school=entry["school"]))

        # Inject curated hostel fee summaries so hostel queries always find the real data
        hostel_curated = doc.get("hostel_curated") or []
        for text in hostel_curated:
            chunks.insert(0, self._chunk(doc, text, 1, curated=True, kind="hostel_fees"))

        if doc["id"] == "srm-admissions-2026-27":
            self._inject_srm_placements(doc, chunks, reader)

    def build_or_load_index(self, force_rebuild=False):
        """Load the precomputed index, or build (train) and save it."""
        if not force_rebuild and os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data["chunks"]
                    self.vectorizer = data["vectorizer"]
                    self.tfidf = data["tfidf"]
                    self.tfidf_norms = data["tfidf_norms"]
                return
            except Exception as e:
                print(f"Failed to load cached index: {e}. Rebuilding...")

        # --- Build from scratch: extract, chunk, curate, train TF-IDF ---
        self.chunks = []
        for doc in self.documents:
            reader, doc_chunks = self._extract_document(doc)
            self._inject_curated_chunks(doc, doc_chunks, reader)
            self.chunks.extend(doc_chunks)

        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            use_idf=True,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self.tfidf = self.vectorizer.fit_transform(texts)
        self.tfidf_norms = np.asarray(self.tfidf.sum(axis=1)).ravel()

        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "vectorizer": self.vectorizer,
                    "tfidf": self.tfidf,
                    "tfidf_norms": self.tfidf_norms,
                    "documents": [
                        {"id": d["id"], "title": d["title"]} for d in self.documents
                    ],
                }, f)
            print(f"Index saved: {len(self.chunks)} chunks -> {self.index_path}")
        except Exception as e:
            print(f"Failed to save index cache: {e}")

    # ----------------------------------------------------------------- search

    @staticmethod
    def _keyword_score(query_tokens, chunk_text):
        """Normalized keyword overlap score (BM25-lite) with word-boundary matching.

        Matching is done on a normalized chunk where punctuation is stripped
        ("b.tech" -> "btech") and tokens must match as whole words, so "tech"
        matches "B.Tech" but NOT "biotechnology" or "technology".
        """
        chunk_norm = re.sub(r'[^a-z0-9 ]', '', chunk_text.lower())
        matches = 0
        for t in query_tokens:
            if not t or len(t) < 2:
                continue
            # Light plural handling in both directions: "admission" matches
            # "admissions", and "fees" also matches "fee".
            variants = {t, t + 's'}
            if t.endswith('s') and len(t) > 3 and not t.endswith(('ss', 'us', 'is')):
                variants.add(t[:-1])
            if any(
                re.search(r'(?<![a-z0-9])' + re.escape(v) + r'(?![a-z0-9])', chunk_norm)
                for v in variants
            ):
                matches += 1
        return matches / max(len(query_tokens), 1)

    LIST_WORDS = {
        "pg", "ug", "postgraduate", "undergraduate", "program", "programs",
        "course", "courses", "degree", "degrees", "list", "school", "schools",
        "offer", "offered", "available",
        "btech", "mtech", "msc", "bsc", "mba", "bba", "mca", "mph", "mcom",
        "bcom", "llm", "phd", "md", "ms", "dm", "mds", "mch", "mpt", "mot",
        "mpharm", "mha",
    }

    def is_list_query(self, query):
        """True when the query looks like a 'show me the list of programs' request."""
        tokens = self._query_tokens(query)
        return sum(1 for t in tokens if t in self.LIST_WORDS) >= 2

    def _query_tokens(self, query):
        """Stopword-filtered query tokens with domain synonym expansion."""
        base = [
            t for t in re.sub(r'[^a-z0-9 ]', '', query.lower()).split()
            if t not in STOPWORDS
        ]
        expanded = []
        for t in base:
            expanded.append(t)
            if t in QUERY_SYNONYMS:
                expanded.extend(QUERY_SYNONYMS[t].split())
        seen = set()
        return [t for t in expanded if not (t in seen or seen.add(t))]

    def search(self, query, top_k=5):
        """Hybrid search: TF-IDF cosine similarity + keyword overlap re-ranking."""
        if self.tfidf is None or len(self.chunks) == 0:
            self.build_or_load_index()

        query_tokens = self._query_tokens(query)
        query_expanded = " ".join(query_tokens)

        # --- TF-IDF cosine similarity (min-max normalized for contrast) ---
        query_vec = self.vectorizer.transform([query_expanded])
        cos_scores = np.asarray((self.tfidf @ query_vec.T).toarray()).ravel()
        query_norm = np.linalg.norm(query_vec.toarray())
        if query_norm > 0 and self.tfidf_norms.max() > 0:
            cos_scores = cos_scores / (self.tfidf_norms * query_norm)
        else:
            cos_scores = np.zeros(len(self.chunks))
        # Min-max normalize for contrast, but ONLY when the query shares real
        # vocabulary with the corpus (raw cosine > 0.02). Otherwise leave zero -
        # this prevents irrelevant queries from manufacturing high scores while
        # still letting TF-IDF idf-weighting break ties between keyword matches.
        cmin, cmax = cos_scores.min(), cos_scores.max()
        if cmax > 0.02 and cmax > cmin:
            cos_scores = (cos_scores - cmin) / (cmax - cmin)
        else:
            cos_scores = np.zeros(len(self.chunks))

        # --- Keyword overlap (raw fraction, boundary-matched) ---
        keyword_scores = np.array([
            self._keyword_score(query_tokens, c["text"]) for c in self.chunks
        ], dtype=float)

        # --- Hybrid score ---
        # Keyword coverage is the primary signal; curated ground-truth chunks are
        # amplified so they beat dense-but-noisy raw fragments on near-ties.
        # TF-IDF cosine only nudges as a tiebreaker (and only when the query
        # shares real vocabulary with the corpus, thanks to the gate above).
        curated = np.array([1.0 if c.get("curated") else 0.0 for c in self.chunks])
        hybrid_scores = (
            1.20 * keyword_scores
            + 0.35 * curated * keyword_scores
            + 0.25 * cos_scores
        )

        # For list-style queries ("pg programs", "all programs", "what degrees"),
        # lift the "Programs Overview" chunk so the complete list leads the answer.
        if self.is_list_query(query):
            overview = np.array([
                1.0 if "Programs Overview" in c["text"] else 0.0
                for c in self.chunks
            ])
            hybrid_scores += 0.25 * overview * keyword_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            {"chunk": self.chunks[idx], "score": float(hybrid_scores[idx])}
            for idx in top_indices
        ]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    engine = RAGEngine()
    engine.build_or_load_index(force_rebuild=True)
    print(f"Index built with {len(engine.chunks)} chunks.\n")
    for q in [
        "all pg programs in srm institute of science and technology",
        "what are the MBA specializations",
        "what are the B.Tech programs",
        "placement statistics and highest salary",
        "what is the admission helpline number",
        "what are the fees for btech",
        "hello",
        "what is the last date for applying",
        "does srm offer hostel accommodation",
        "what is the dress code",
    ]:
        print(f"==== QUERY: {q}")
        for r in engine.search(q, top_k=3):
            print(f"  [score {r['score']:.3f} | page {r['chunk']['page']}] {r['chunk']['text'][:100]}...")
        print()
