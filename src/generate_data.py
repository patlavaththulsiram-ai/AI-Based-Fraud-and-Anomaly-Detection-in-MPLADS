"""Synthetic MPLADS Dataset Generator with Controlled Anomaly Injections
Creates ~5,000 realistic project records across 12 Indian states with realistic
public works categories, financial structures, timelines, and labelled anomalies (~8-12%).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config import Config

# Reproducibility seed
SEED = Config.RANDOM_STATE
random.seed(SEED)
np.random.seed(SEED)

STATES_DATA = {
    "Telangana": {
        "districts": ["Hyderabad", "Ranga Reddy", "Medchal-Malkajgiri", "Nizamabad", "Karimnagar", "Warangal", "Khammam", "Nalgonda", "Mahabubnagar", "Siddipet"],
        "constituencies": ["Hyderabad", "Secunderabad", "Malkajgiri", "Chevella", "Nizamabad", "Karimnagar", "Warangal", "Khammam", "Nalgonda", "Medak"],
        "mps": ["Asaduddin Owaisi", "G. Kishan Reddy", "Etela Rajender", "Konda Vishweshwar Reddy", "D. Arvind", "Bandi Sanjay Kumar", "Kadiyam Kavya", "Ramasahayam Raghuram Reddy", "K. Raghuveer Reddy", "M. Raghunandan Rao"],
        "lat_range": (17.1, 18.9),
        "lon_range": (78.1, 79.9)
    },
    "Andhra Pradesh": {
        "districts": ["Visakhapatnam", "Krishna", "Guntur", "Tirupati", "Kurnool", "Nellore", "East Godavari", "West Godavari", "Anantapur", "YSR Kadapa"],
        "constituencies": ["Visakhapatnam", "Machilipatnam", "Guntur", "Tirupati", "Kurnool", "Nellore", "Rajahmundry", "Eluru", "Anantapur", "Kadapa"],
        "mps": ["M. Sribharat", "Balashowry Vallabhaneni", "Dr. Pemmasani Chandra Sekhar", "M. Gurumoorthy", "Bastipati Nagaraju Panchalingala", "Vemireddy Prabhakar Reddy", "Daggubati Purandeswari", "Putta Mahesh Kumar", "Ambica G. Lakshminarayana", "Y. S. Avinash Reddy"],
        "lat_range": (14.2, 17.8),
        "lon_range": (79.5, 83.3)
    },
    "Maharashtra": {
        "districts": ["Mumbai City", "Mumbai Suburban", "Pune", "Nagpur", "Nashik", "Thane", "Chhatrapati Sambhajinagar", "Solapur", "Kolhapur", "Amravati"],
        "constituencies": ["Mumbai South", "Mumbai North", "Pune", "Nagpur", "Nashik", "Thane", "Aurangabad", "Solapur", "Kolhapur", "Amravati"],
        "mps": ["Arvind Sawant", "Piyush Goyal", "Murlidhar Mohol", "Nitin Gadkari", "Rajabhau Waje", "Naresh Mhaske", "Sandipan Bhumre", "Praniti Shinde", "Shahu Chhatrapati", "Balwant Wankhade"],
        "lat_range": (18.5, 21.1),
        "lon_range": (72.8, 79.1)
    },
    "Karnataka": {
        "districts": ["Bengaluru Urban", "Bengaluru Rural", "Mysuru", "Dharwad", "Dakshina Kannada", "Belagavi", "Kalaburagi", "Shivamogga", "Tumakuru", "Ballari"],
        "constituencies": ["Bangalore South", "Bangalore Rural", "Mysore", "Dharwad", "Dakshina Kannada", "Belgaum", "Gulbarga", "Shimoga", "Tumkur", "Bellary"],
        "mps": ["Tejasvi Surya", "Dr. C. N. Manjunath", "Yaduveer Krishnadatta Chamaraja Wadiyar", "Pralhad Joshi", "Capt. Brijesh Chowta", "Jagadish Shettar", "Radhakrishna Doddamani", "B. Y. Raghavendra", "V. Somanna", "E. Tukaram"],
        "lat_range": (12.8, 16.5),
        "lon_range": (74.8, 77.6)
    },
    "Tamil Nadu": {
        "districts": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Vellore", "Erode", "Thanjavur", "Kanyakumari"],
        "constituencies": ["Chennai South", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem", "Tirunelveli", "Vellore", "Erode", "Thanjavur", "Kanyakumari"],
        "mps": ["Thamizhachi Thangapandian", "Ganapathi P. Rajkumar", "Su. Venkatesan", "Durai Vaiko", "T. M. Selvaganapathi", "Robert Bruce", "D. M. Kathir Anand", "K. E. Prakash", "S. Murasoli", "Vijay Vasanth"],
        "lat_range": (8.1, 13.1),
        "lon_range": (77.2, 80.3)
    },
    "Kerala": {
        "districts": ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kollam", "Palakkad", "Kannur", "Alappuzha", "Kottayam", "Malappuram"],
        "constituencies": ["Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Kollam", "Palakkad", "Kannur", "Alappuzha", "Kottayam", "Malappuram"],
        "mps": ["Shashi Tharoor", "Hibi Eden", "M. K. Raghavan", "Suresh Gopi", "N. K. Premachandran", "V. K. Sreekandan", "K. Sudhakaran", "K. C. Venugopal", "Francis George", "E. T. Mohammed Basheer"],
        "lat_range": (8.5, 11.9),
        "lon_range": (75.5, 77.0)
    },
    "Gujarat": {
        "districts": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Junagadh", "Anand", "Kutch"],
        "constituencies": ["Ahmedabad East", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Junagadh", "Anand", "Kachchh"],
        "mps": ["Hasmukh Patel", "Mukesh Dalal", "Hemang Joshi", "Parshottam Rupala", "Nimuben Bambhaniya", "Poonamben Maadam", "Amit Shah", "Rajesh Chudasama", "Mitesh Patel", "Vinodbhai Chavda"],
        "lat_range": (21.1, 23.4),
        "lon_range": (70.1, 73.2)
    },
    "Rajasthan": {
        "districts": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer", "Bhilwara", "Alwar", "Sikar", "Barmer"],
        "constituencies": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer", "Bhilwara", "Alwar", "Sikar", "Barmer"],
        "mps": ["Manju Sharma", "Gajendra Singh Shekhawat", "Manna Lal Rawat", "Om Birla", "Arjun Ram Meghwal", "Bhagirath Choudhary", "Damodar Agarwal", "Bhupender Yadav", "Amra Ram", "Ummeda Ram Beniwal"],
        "lat_range": (24.5, 28.1),
        "lon_range": (71.3, 76.8)
    },
    "Uttar Pradesh": {
        "districts": ["Lucknow", "Varanasi", "Kanpur Nagar", "Agra", "Prayagraj", "Meerut", "Ghaziabad", "Gorakhpur", "Bareilly", "Ayodhya"],
        "constituencies": ["Lucknow", "Varanasi", "Kanpur", "Agra", "Allahabad", "Meerut", "Ghaziabad", "Gorakhpur", "Bareilly", "Faizabad"],
        "mps": ["Rajnath Singh", "Narendra Modi", "Ramesh Awasthi", "S. P. Singh Baghel", "Ujjwal Raman Singh", "Arun Govil", "Atul Garg", "Ravi Kishan", "Chhatrapal Singh Gangwar", "Awadhesh Prasad"],
        "lat_range": (25.3, 28.7),
        "lon_range": (77.4, 83.2)
    },
    "West Bengal": {
        "districts": ["Kolkata", "Howrah", "North 24 Parganas", "South 24 Parganas", "Darjeeling", "Paschim Bardhaman", "Murshidabad", "Nadia", "Hooghly", "Malda"],
        "constituencies": ["Kolkata Dakshin", "Howrah", "Dum Dum", "Diamond Harbour", "Darjeeling", "Asansol", "Baharampur", "Krishnanagar", "Hooghly", "Maldaha Uttar"],
        "mps": ["Mala Roy", "Prasun Banerjee", "Saugata Roy", "Abhishek Banerjee", "Raju Bista", "Shatrughan Sinha", "Yusuf Pathan", "Mahua Moitra", "Rachna Banerjee", "Khagen Murmu"],
        "lat_range": (22.2, 26.8),
        "lon_range": (87.8, 88.9)
    },
    "Odisha": {
        "districts": ["Khordha", "Cuttack", "Puri", "Sundargarh", "Ganjam", "Sambalpur", "Balasore", "Mayurbhanj", "Koraput", "Angul"],
        "constituencies": ["Bhubaneswar", "Cuttack", "Puri", "Sundargarh", "Aska", "Sambalpur", "Balasore", "Mayurbhanj", "Koraput", "Dhenkanal"],
        "mps": ["Aparajita Sarangi", "Bhartruhari Mahtab", "Sambit Patra", "Jual Oram", "Anita Subhadarshini", "Dharmendra Pradhan", "Pratap Chandra Sarangi", "Naba Charan Majhi", "Saptagiri Sankar Ulaka", "Rudra Narayan Pany"],
        "lat_range": (18.8, 22.1),
        "lon_range": (82.7, 86.9)
    },
    "Madhya Pradesh": {
        "districts": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Dewas", "Satna", "Ratlam", "Rewa"],
        "constituencies": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Dewas", "Satna", "Ratlam", "Rewa"],
        "mps": ["Alok Sharma", "Shankar Lalwani", "Bharat Singh Kushwah", "Ashish Dubey", "Anil Firojiya", "Lata Wankhede", "Mahendra Singh Solanki", "Ganesh Singh", "Anita Nagar Singh Chouhan", "Janardan Mishra"],
        "lat_range": (22.7, 26.2),
        "lon_range": (75.8, 81.3)
    }
}

AGENCIES = [
    "Public Works Department (PWD)",
    "Panchayati Raj Engineering Dept (PRED)",
    "Municipal Corporation",
    "Rural Water Supply & Sanitation (RWSS)",
    "State Irrigation Department",
    "District Rural Development Agency (DRDA)"
]

CATEGORIES = [
    "Education",
    "Health",
    "Roads",
    "Water",
    "Electricity",
    "Community Assets"
]

WORK_TEMPLATES = {
    "Education": [
        "Construction of 2 additional smart classrooms at ZP High School, {loc}",
        "Renovation and roof waterproofing of Government Primary School building in {loc}",
        "Setting up of 25-terminal computer and digital skills lab at Govt Model College, {loc}",
        "Supply of ergonomic dual desks and library furniture to Zilla Parishad School, {loc}",
        "Construction of separate hygienic sanitation blocks for girls at Govt High School, {loc}",
        "Installation of 10 kW rooftop solar energy plant at Government ITI, {loc}"
    ],
    "Health": [
        "Installation of 500 LPM PSA Medical Oxygen Generation Plant at Area Hospital, {loc}",
        "Procurement of 4 modern hemodialysis units and patient monitors for Sub-District Hospital, {loc}",
        "Construction of maternal and child primary healthcare sub-centre building in {loc}",
        "Supply of advanced cardiac life support (ACLS) mobile ambulance unit for {loc}",
        "Civil renovation and diagnostic equipment installation at Community Health Centre, {loc}",
        "Setting up of fully automated pathology diagnostic laboratory at Taluk Hospital, {loc}"
    ],
    "Roads": [
        "Laying of CC (Cement Concrete) pavement road from main bus route to Ward 8, {loc}",
        "Widening and bitumen blacktopping of agricultural market connectivity road in {loc}",
        "Construction of 4-span RCC box culvert across stormwater stream at {loc}",
        "Upgradation of internal village link road with side drains and street signage in {loc}",
        "Reconstruction of flood-damaged culvert and approach road embankment near {loc}",
        "Laying of interlocking concrete paver road around primary market yard, {loc}"
    ],
    "Water": [
        "Drilling of deep borewell and erection of solar-powered mini piped water scheme in {loc}",
        "Laying of 3.5 km HDPE water supply distribution network and 60,000L OHSR at {loc}",
        "Installation of 2,000 LPH community Reverse Osmosis (RO) water purification plant at {loc}",
        "Rejuvenation, silt removal, and embankment stone pitching of traditional percolation tank at {loc}",
        "Replacement of damaged CI feeder pipeline and repair of community water sumps in {loc}",
        "Construction of integrated rainwater harvesting and ground aquifer recharge structures at {loc}"
    ],
    "Electricity": [
        "Installation of 80 integrated solar LED street lighting units along primary bypass in {loc}",
        "Erection of 100 kVA distribution transformer and HT/LT transmission line augmentation at {loc}",
        "Installation of 30-meter high-mast lighting mast with 8 LED floodlights at public junction, {loc}",
        "Setting up of 20 kW decentralized grid-tied solar photovoltaic system at Gram Panchayat, {loc}",
        "Replacement of overhead bare conductors with insulated aerial bunched cables in {loc}",
        "Installation of energy-efficient smart LED street lighting network with timer controls at {loc}"
    ],
    "Community Assets": [
        "Construction of multipurpose community convention hall for social ceremonies in {loc}",
        "Development of open-air gymnasium and 400m walking track in public town park, {loc}",
        "Construction of public library and competitive examination study centre at {loc}",
        "Construction of modern passenger transit shelter with digital timetable display at {loc}",
        "Construction of boundary wall and flood protection drainage around public cremation ground, {loc}",
        "Development of farmer weekly vegetable market shed with concrete raised platforms at {loc}"
    ]
}

def generate_mplads_dataset(num_records: int = 5000, anomaly_ratio: float = 0.10) -> pd.DataFrame:
    """
    Generates realistic synthetic MPLADS records with controlled anomaly injections.
    
    Returns:
        pd.DataFrame with canonical columns + is_anomaly + anomaly_type
    """
    num_anomalies = int(num_records * anomaly_ratio)
    num_normal = num_records - num_anomalies

    records = []
    
    # State list for uniform distribution
    state_keys = list(STATES_DATA.keys())
    
    # Anomaly categories distribution:
    # 1. cost_overrun (expenditure 20-80% above estimate)
    # 2. expenditure_exceeds_sanction (expenditure > sanctioned)
    # 3. premature_completion (completed with progress < 60%)
    # 4. chronic_delay (delayed > 300 days or delayed with low progress)
    # 5. near_duplicate (near duplicate description in same district)
    # 6. suspicious_round_amount (exact 5, 10, 25 or 49.99 Lakhs)
    # 7. invalid_values (negative cost, end date before sanction date, progress > 100%)
    
    anomaly_types = [
        "cost_overrun",
        "expenditure_exceeds_sanction",
        "premature_completion",
        "chronic_delay",
        "near_duplicate",
        "suspicious_round_amount",
        "invalid_values"
    ]
    
    # Ensure near_duplicate is even
    near_dup = int(num_anomalies * 0.12)
    if near_dup % 2 != 0:
        near_dup += 1

    anomaly_counts = {
        "cost_overrun": int(num_anomalies * 0.18),
        "expenditure_exceeds_sanction": int(num_anomalies * 0.18),
        "premature_completion": int(num_anomalies * 0.14),
        "chronic_delay": int(num_anomalies * 0.16),
        "near_duplicate": near_dup,
        "suspicious_round_amount": int(num_anomalies * 0.12),
        "invalid_values": int(num_anomalies * 0.10)
    }
    
    # Balance discrepancy strictly
    assigned = sum(anomaly_counts.values())
    if assigned < num_anomalies:
        anomaly_counts["cost_overrun"] += (num_anomalies - assigned)
    elif assigned > num_anomalies:
        anomaly_counts["cost_overrun"] -= (assigned - num_anomalies)

    # Base reference date for timeline calculations (Current Simulated Date: Late 2025/2026)
    today = datetime(2026, 3, 1)

    print(f"Generating {num_records} MPLADS projects ({num_normal} normal, {num_anomalies} anomalies)...")

    # Helper to create a single base realistic record
    def make_base_record(idx: int, state_name: str = None):
        if not state_name:
            state_name = random.choice(state_keys)
        s_data = STATES_DATA[state_name]
        
        loc_idx = random.randint(0, len(s_data["districts"]) - 1)
        district = s_data["districts"][loc_idx]
        constituency = s_data["constituencies"][loc_idx]
        mp = s_data["mps"][loc_idx]
        
        agency = random.choice(AGENCIES)
        category = random.choice(CATEGORIES)
        
        loc_name = f"{district} Block-{random.randint(1, 9)}"
        template = random.choice(WORK_TEMPLATES[category])
        desc = template.format(loc=loc_name)
        
        # Financial estimates (in INR Lakhs: 3.00 to 75.00 Lakhs typical for MPLADS works)
        estimated_cost = round(random.uniform(4.0, 65.0), 2)
        # Sanctioned amount normally 95% to 100% of estimate
        sanctioned_amount = round(estimated_cost * random.uniform(0.95, 1.0), 2)
        # Authorized released funds normally 80% to 100% of sanctioned
        authorized_amount = round(sanctioned_amount * random.uniform(0.80, 1.0), 2)
        
        # Dates
        # Sanctions spread from 2021 to 2025
        sanction_days_ago = random.randint(120, 1500)
        sanction_date = today - timedelta(days=sanction_days_ago)
        
        # Expected duration: 180 to 540 days (6 to 18 months)
        planned_duration = random.randint(180, 540)
        expected_completion = sanction_date + timedelta(days=planned_duration)
        
        # Default normal status logic
        if expected_completion < today:
            # Should normally be completed or near completion
            rand_roll = random.random()
            if rand_roll < 0.75:
                status = "Completed"
                progress = 100.0
                actual_completion = expected_completion + timedelta(days=random.randint(-30, 60))
                if actual_completion > today:
                    actual_completion = today - timedelta(days=random.randint(5, 45))
                # Expenditure roughly 92% to 99% of sanctioned
                expenditure = round(sanctioned_amount * random.uniform(0.92, 0.99), 2)
            elif rand_roll < 0.92:
                status = "Ongoing"
                progress = round(random.uniform(70.0, 95.0), 1)
                actual_completion = None
                expenditure = round(authorized_amount * (progress / 100.0) * random.uniform(0.90, 1.0), 2)
            else:
                status = "Delayed"
                progress = round(random.uniform(40.0, 75.0), 1)
                actual_completion = None
                expenditure = round(authorized_amount * (progress / 100.0) * random.uniform(0.85, 1.02), 2)
        else:
            # Future expected completion
            rand_roll = random.random()
            if rand_roll < 0.15:
                status = "Pending"
                progress = 0.0
                actual_completion = None
                expenditure = 0.0
            else:
                status = "Ongoing"
                elapsed = (today - sanction_date).days
                expected_ratio = min(1.0, max(0.1, elapsed / planned_duration))
                progress = round(min(95.0, expected_ratio * 100 * random.uniform(0.8, 1.1)), 1)
                actual_completion = None
                expenditure = round(authorized_amount * (progress / 100.0) * random.uniform(0.85, 1.0), 2)

        # Ensure expenditure <= authorized for normal records
        expenditure = min(expenditure, authorized_amount)
        
        # Coordinates
        lat = round(random.uniform(s_data["lat_range"][0], s_data["lat_range"][1]), 5)
        lon = round(random.uniform(s_data["lon_range"][0], s_data["lon_range"][1]), 5)
        
        state_code = state_name[:2].upper()
        dist_code = district[:3].upper()
        year = sanction_date.year
        work_id = f"MPLAD-{state_code}-{dist_code}-{year}-{idx:05d}"
        
        return {
            "work_id": work_id,
            "mp_name": mp,
            "constituency": constituency,
            "state": state_name,
            "district": district,
            "implementing_agency": agency,
            "work_category": category,
            "work_description": desc,
            "sanctioned_amount": sanctioned_amount,
            "estimated_cost": estimated_cost,
            "authorized_amount": authorized_amount,
            "expenditure": expenditure,
            "sanction_date": sanction_date.strftime("%Y-%m-%d"),
            "expected_completion_date": expected_completion.strftime("%Y-%m-%d"),
            "actual_completion_date": actual_completion.strftime("%Y-%m-%d") if actual_completion else "",
            "status": status,
            "progress_percent": progress,
            "latitude": lat,
            "longitude": lon,
            "is_anomaly": 0,
            "anomaly_type": "None"
        }

    # 1. Generate normal records
    global_id = 1
    for _ in range(num_normal):
        records.append(make_base_record(global_id))
        global_id += 1

    # 2. Inject Anomaly: Cost Overrun (Expenditure 20-80% above estimate)
    for _ in range(anomaly_counts["cost_overrun"]):
        rec = make_base_record(global_id)
        global_id += 1
        overrun_factor = random.uniform(1.25, 1.80)
        rec["expenditure"] = round(rec["estimated_cost"] * overrun_factor, 2)
        # In actual corrupt scenarios, authorized amount may have been inflated or overdrawn
        rec["authorized_amount"] = max(rec["authorized_amount"], rec["expenditure"])
        rec["sanctioned_amount"] = round(rec["estimated_cost"] * 1.05, 2) # sanction didn't cover overrun
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "cost_overrun"
        records.append(rec)

    # 3. Inject Anomaly: Expenditure Exceeds Sanctioned Amount
    for _ in range(anomaly_counts["expenditure_exceeds_sanction"]):
        rec = make_base_record(global_id)
        global_id += 1
        rec["expenditure"] = round(rec["sanctioned_amount"] * random.uniform(1.15, 1.55), 2)
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "expenditure_exceeds_sanction"
        records.append(rec)

    # 4. Inject Anomaly: Premature Completion Claim (Completed with progress < 60%)
    for _ in range(anomaly_counts["premature_completion"]):
        rec = make_base_record(global_id)
        global_id += 1
        rec["status"] = "Completed"
        rec["progress_percent"] = round(random.uniform(12.0, 52.0), 1)
        rec["actual_completion_date"] = (today - timedelta(days=random.randint(20, 180))).strftime("%Y-%m-%d")
        rec["expenditure"] = round(rec["sanctioned_amount"] * random.uniform(0.90, 1.0), 2)
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "premature_completion"
        records.append(rec)

    # 5. Inject Anomaly: Chronic Delay / Prolonged Inaction
    for _ in range(anomaly_counts["chronic_delay"]):
        rec = make_base_record(global_id)
        global_id += 1
        # Set sanction date far back (3 to 4 years ago)
        s_date = today - timedelta(days=random.randint(1100, 1600))
        rec["sanction_date"] = s_date.strftime("%Y-%m-%d")
        rec["expected_completion_date"] = (s_date + timedelta(days=365)).strftime("%Y-%m-%d")
        rec["actual_completion_date"] = ""
        rec["status"] = "Delayed"
        rec["progress_percent"] = round(random.uniform(5.0, 35.0), 1)
        # Big expenditure drain despite poor progress
        rec["expenditure"] = round(rec["authorized_amount"] * random.uniform(0.70, 0.95), 2)
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "chronic_delay"
        records.append(rec)

    # 6. Inject Anomaly: Near-Duplicate Work Descriptions (same district & category)
    dup_pairs = anomaly_counts["near_duplicate"] // 2
    for _ in range(dup_pairs):
        rec_orig = make_base_record(global_id)
        global_id += 1
        # Create twin record with high lexical similarity in same district & category
        rec_twin = make_base_record(global_id, state_name=rec_orig["state"])
        global_id += 1
        rec_twin["district"] = rec_orig["district"]
        rec_twin["constituency"] = rec_orig["constituency"]
        rec_twin["mp_name"] = rec_orig["mp_name"]
        rec_twin["work_category"] = rec_orig["work_category"]
        
        # Slightly alter the phrasing or location name to make TF-IDF similarity ~ 0.85-0.95
        orig_desc = rec_orig["work_description"]
        if "smart classrooms" in orig_desc:
            twin_desc = orig_desc.replace("smart classrooms", "modern digital classrooms")
        elif "CC (Cement Concrete)" in orig_desc:
            twin_desc = orig_desc.replace("CC (Cement Concrete) pavement road", "Cement Concrete road pavement")
        elif "Reverse Osmosis (RO)" in orig_desc:
            twin_desc = orig_desc.replace("Reverse Osmosis (RO)", "community RO drinking water plant")
        elif "solar LED street lighting" in orig_desc:
            twin_desc = orig_desc.replace("solar LED street lighting units", "integrated solar street lights")
        elif "PSA Medical Oxygen" in orig_desc:
            twin_desc = orig_desc.replace("PSA Medical Oxygen Generation Plant", "Medical Oxygen generation plant unit")
        else:
            twin_desc = f"{orig_desc} - Phase II extension"
            
        rec_twin["work_description"] = twin_desc
        rec_twin["sanctioned_amount"] = round(rec_orig["sanctioned_amount"] * random.uniform(0.96, 1.04), 2)
        rec_twin["estimated_cost"] = rec_orig["estimated_cost"]
        rec_twin["authorized_amount"] = rec_twin["sanctioned_amount"]
        rec_twin["expenditure"] = round(rec_twin["sanctioned_amount"] * 0.95, 2)
        
        rec_orig["is_anomaly"] = 1
        rec_orig["anomaly_type"] = "near_duplicate"
        rec_twin["is_anomaly"] = 1
        rec_twin["anomaly_type"] = "near_duplicate"
        
        records.append(rec_orig)
        records.append(rec_twin)

    # 7. Inject Anomaly: Suspicious Round / Just-Below Threshold Disbursal
    round_thresholds = [5.0, 10.0, 25.0, 50.0]
    below_thresholds = [4.99, 9.98, 24.95, 49.90]
    for _ in range(anomaly_counts["suspicious_round_amount"]):
        rec = make_base_record(global_id)
        global_id += 1
        if random.random() < 0.5:
            target_amt = random.choice(round_thresholds)
        else:
            target_amt = random.choice(below_thresholds)
        rec["sanctioned_amount"] = target_amt
        rec["estimated_cost"] = target_amt
        rec["authorized_amount"] = target_amt
        rec["expenditure"] = target_amt
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "suspicious_round_amount"
        records.append(rec)

    # 8. Inject Anomaly: Invalid / Corrupt Values
    for _ in range(anomaly_counts["invalid_values"]):
        rec = make_base_record(global_id)
        global_id += 1
        invalid_mode = random.choice(["negative_cost", "reverse_dates", "excess_progress"])
        if invalid_mode == "negative_cost":
            rec["expenditure"] = -abs(round(random.uniform(3.5, 25.0), 2))
            rec["estimated_cost"] = -abs(rec["estimated_cost"])
        elif invalid_mode == "reverse_dates":
            # expected_completion_date is BEFORE sanction_date
            s_date = datetime.strptime(rec["sanction_date"], "%Y-%m-%d")
            rec["expected_completion_date"] = (s_date - timedelta(days=random.randint(60, 180))).strftime("%Y-%m-%d")
        else:
            rec["progress_percent"] = round(random.uniform(130.0, 210.0), 1)
        rec["is_anomaly"] = 1
        rec["anomaly_type"] = "invalid_values"
        records.append(rec)

    # Convert to DataFrame and shuffle
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    
    # Save to data/synthetic
    out_path = Config.DATA_SYNTHETIC_DIR / f"mplads_synthetic_{len(df)}.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved successfully to: {out_path} ({len(df)} rows)")
    print(f"Anomaly breakdown:\n{df['anomaly_type'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_mplads_dataset(5000, 0.10)
