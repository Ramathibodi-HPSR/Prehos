# Prehos

## Overview

**Prehos** is a research project by the Health Policy and Systems Research (HPSR) Unit at the Faculty of Medicine Ramathibodi Hospital, Mahidol University, Thailand. It aims to evaluate the effects of decentralization on the quality and accessibility of emergency medical services (EMS) in Thailand using both operational data and patient survey evidence.

## Repository Contents

- `main.ipynb`: This notebook contains the primary analysis for the research article titled *"The Impact of Decentralization Policy on Accessibility and Quality of Emergency Medical Services in Thailand"*. It uses over 3.3 million EMS operation records from the national ITEMS database to evaluate differences in performance metrics (triage accuracy, time-sensitive indicators, and OHCA survival) between centralized and decentralized dispatch models. The data underlying this article cannot be shared publicly due to ethical issue but can be requested from the contact provided. 
- `survey.ipynb`: This notebook provides analysis for the study published in the Journal of Health Systems Research (HSRI). It presents patient perspectives on EMS quality under decentralized governance, based on a structured survey across multiple provinces, focusing on service satisfaction, trust, and service utilization in emergency situations.
- `scripts/`: Utility scripts for preprocessing, cleaning, and modeling.

## Technologies Used

- **Python**: For data analysis and modeling.
- **Jupyter Notebook**: For interactive exploration and reporting.
- **Pandas, NumPy, Scikit-learn**: For statistical computation and imputation.
- **Tableau**: Used in survey analysis for visualizing patient response metrics.

## Getting Started

To run this project locally:

```bash
git clone https://github.com/Ramathibodi-HPSR/Prehos.git
cd Prehos
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook
```

## Key Findings

- **From main.ipynb**: Decentralization of EMS dispatch centres did not significantly alter survival outcomes for OHCA patients, suggesting non-inferiority compared to the centralized model. However, over-triage and under-triage were more frequent in decentralized settings.
- **From survey.ipynb**: Public confidence and EMS utilization rates were statistically lower in provinces with decentralized EMS governance, although satisfaction levels remained comparable.

## Contributing

We welcome contributions and discussions to improve the project. Please fork and create a pull request or open an issue to suggest changes.

## References
1. Sitthirat P, Suppawittaya P, Limchantra P, Kaewkamjornchai P, Atiksawedparit P, Suriyawongpaisal P, et al. มุมมองผู้ป่วยเกี่ยวกับคุณภาพบริการการแพทย์ฉุกเฉินและปัจจัยที่เกี่ยวข้องในบริบทการกระจายอำนาจสู่องค์กรปกครองส่วนท้องถิ่น. Journal of Health Systems Research. 2024 Dec 29;18(4):442–58. [link](https://kb.hsri.or.th/dspace/handle/11228/6212?src=%2Fdspace%2Fhandle%2F11228%2F1%2Fdiscover%3Frpp%3D10%26etal%3D0%26group_by%3Dnone%26page%3D6%26filtertype_0%3Dauthor%26filter_relational_operator_0%3Dequals%26filter_0%3D%25E0%25B8%25AA%25E0%25B8%25B1%25E0%25B8%25A1%25E0%25B8%25A4%25E0%25B8%2597%25E0%25B8%2598%25E0%25B8%25B4%25E0%25B9%258C%2520%25E0%25B8%25A8%25E0%25B8%25A3%25E0%25B8%25B5%25E0%25B8%2598%25E0%25B8%25B3%25E0%25B8%25A3%25E0%25B8%2587%25E0%25B8%25AA%25E0%25B8%25A7%25E0%25B8%25B1%25E0%25B8%25AA%25E0%25B8%2594%25E0%25B8%25B4%25E0%25B9%258C%26locale-attribute%3Dth&offset=9&locale-attribute=th)

## License

This repository is licensed under the MIT License.

## Contact

For inquiries, please contact the HPSR Unit:

- **Email**: [peerasit.sit@mahidol.ac.th](mailto:peerasit.sit@mahidol.ac.th)