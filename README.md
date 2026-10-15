# HR Analytics Dashboard

An end-to-end HR analytics portfolio project built with Python, Pandas, Streamlit, Plotly, and scikit-learn.

## Features
- KPI cards for headcount, attrition rate, average monthly income, and job satisfaction
- Interactive filters for department, attrition status, and age
- Charts for headcount, attrition by department, income vs. satisfaction, and overtime
- Random Forest classification demonstration for employee attrition
- Model metrics: accuracy, precision, recall, and F1 score
- Sample prediction form and filtered CSV download

## Dataset
`hr_employee_data.csv` contains 600 **synthetic** employee records created for this portfolio project. It does not contain real employee data. Use it for learning and demonstration only.

## Run locally
Python 3.10–3.12 is recommended for broad package compatibility.

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py` and `hr_employee_data.csv` in the same folder.

## GitHub repository description
Interactive HR analytics dashboard built with Streamlit and Python, including workforce KPIs, attrition analysis, and a machine-learning demonstration using synthetic employee data.

## Upload to GitHub
Create a repository named `hr-analytics-dashboard`, then run:

```bash
git init
git add .
git commit -m "Build HR analytics dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/hr-analytics-dashboard.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Responsible use
The model is a learning demonstration trained on synthetic data. Model estimates can reflect data bias and should not be used to rank, discipline, hire, or fire real employees.
