import os
import re
import sys
import urllib.request
from bs4 import BeautifulSoup

URLS = [
    ("web_srm_research_and_patents.txt", "https://www.srmist.edu.in/research/", "SRMIST Research, Grants & Patents"),
    ("web_srm_incubation_siic.txt", "https://www.srmist.edu.in/directorate-of-innovation-and-incubation/", "SRMIST Directorate of Innovation & Incubation (SIIC)"),
    ("web_srm_international_relations.txt", "https://www.srmist.edu.in/international-relations/", "SRMIST Directorate of International Relations"),
    ("web_srm_semester_abroad_program.txt", "https://www.srmist.edu.in/international-relations/semester-abroad-program/", "SRMIST Semester Abroad Program (SAP)"),
    ("web_srm_central_library.txt", "https://www.srmist.edu.in/library/", "SRMIST Central Library & Digital Resources"),
    ("web_srm_controller_of_examinations.txt", "https://www.srmist.edu.in/coe/", "SRMIST Controller of Examinations (CoE)"),
    ("web_srm_transport_and_buses.txt", "https://www.srmist.edu.in/facilities/transport/", "SRMIST Transport Services & Bus Routes"),
    ("web_srm_hostel_facilities.txt", "https://www.srmist.edu.in/hostel/", "SRMIST Hostels & Student Accommodation"),
    ("web_srm_medical_hospital_facility.txt", "https://www.srmist.edu.in/facility/medical-facility/", "SRMIST Medical College Hospital & Health Facilities"),
    ("web_srm_emergency_contact_directory.txt", "https://www.srmist.edu.in/contact-us/", "SRMIST Official Contacts & Helpline Directory"),
    ("web_srm_sports_and_athletics.txt", "https://www.srmist.edu.in/sports/", "SRMIST Directorate of Sports & Athletics"),
    ("web_srm_student_affairs_clubs.txt", "https://www.srmist.edu.in/student-affairs/", "SRMIST Directorate of Student Affairs & Campus Clubs"),
    ("web_srm_placements_overview.txt", "https://www.srmist.edu.in/placements/", "SRMIST Placements & Career Development"),
    ("web_srm_admissions_overview.txt", "https://www.srmist.edu.in/admission-india/", "SRMIST Admissions Overview"),
]

def clean_html(html, title, url):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
    
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Remove repeated short menu items
    cleaned_lines = []
    for line in lines:
        if len(line) < 3 and not line.isalnum():
            continue
        cleaned_lines.append(line)
    
    content = "\n".join(cleaned_lines)
    header = f"=== DOCUMENT TITLE: {title} ===\n=== SOURCE URL: {url} ===\n\n"
    return header + content

def main():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    scraped_count = 0
    
    for filename, url, title in URLS:
        print(f"Scraping [{title}] from {url}...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode('utf-8', errors='ignore')
                clean_text = clean_html(html, title, url)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                print(f"  -> Saved {len(clean_text)} characters to {filename}")
                scraped_count += 1
        except Exception as e:
            print(f"  -> Failed to scrape {url}: {e}")
            
    print(f"\nDone! Scraped {scraped_count}/{len(URLS)} web pages successfully.")

if __name__ == "__main__":
    main()
