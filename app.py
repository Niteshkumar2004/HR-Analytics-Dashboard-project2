import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="HR Analytics Dashboard", page_icon="👥", layout="wide")
st.title("👥 HR Analytics Dashboard")
st.caption("Explore workforce trends, understand employee attrition, and support data-informed HR decisions.")

@st.cache_data
def load_data():
    return pd.read_csv("hr_employee_data.csv")

df = load_data()
required = {"EmployeeID", "Department", "MonthlyIncome", "JobSatisfaction", "Attrition"}
if not required.issubset(df.columns):
    st.error("The dataset is missing required columns. Please use the included hr_employee_data.csv file.")
    st.stop()

with st.sidebar:
    st.header("Dashboard filters")
    dept_options = sorted(df["Department"].dropna().unique().tolist())
    selected_depts = st.multiselect("Department", dept_options, default=dept_options)
    attrition_options = sorted(df["Attrition"].dropna().unique().tolist())
    selected_attrition = st.multiselect("Attrition", attrition_options, default=attrition_options)
    age_range = st.slider("Age range", int(df["Age"].min()), int(df["Age"].max()),
                          (int(df["Age"].min()), int(df["Age"].max())))

filtered = df[
    df["Department"].isin(selected_depts) &
    df["Attrition"].isin(selected_attrition) &
    df["Age"].between(age_range[0], age_range[1])
]
if filtered.empty:
    st.warning("No employees match the selected filters. Change the filters to view results.")
    st.stop()

total = len(filtered)
attrition_rate = filtered["Attrition"].eq("Yes").mean() * 100
avg_income = filtered["MonthlyIncome"].mean()
avg_satisfaction = filtered["JobSatisfaction"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", f"{total:,}")
c2.metric("Attrition rate", f"{attrition_rate:.1f}%")
c3.metric("Average monthly income", f"₹{avg_income:,.0f}")
c4.metric("Average job satisfaction", f"{avg_satisfaction:.1f}/5")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Employees by department")
    counts = filtered.groupby("Department", as_index=False).size().rename(columns={"size": "Employees"})
    st.plotly_chart(px.bar(counts, x="Department", y="Employees", color="Department",
                           title="Headcount by department"), use_container_width=True)
with right:
    st.subheader("Attrition by department")
    attr = filtered.groupby("Department", as_index=False).agg(
        Employees=("EmployeeID", "count"),
        AttritionRate=("Attrition", lambda x: (x == "Yes").mean() * 100))
    st.plotly_chart(px.bar(attr, x="Department", y="AttritionRate", color="Department",
                           labels={"AttritionRate": "Attrition rate (%)"},
                           title="Share of employees who left"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Income and satisfaction")
    st.plotly_chart(px.scatter(filtered, x="MonthlyIncome", y="JobSatisfaction",
        color="Attrition", hover_data=["Department", "JobLevel", "YearsAtCompany"],
        labels={"MonthlyIncome": "Monthly income (₹)", "JobSatisfaction": "Job satisfaction"},
        title="Income vs job satisfaction"), use_container_width=True)
with right:
    st.subheader("Overtime and attrition")
    ot = filtered.groupby(["OverTime", "Attrition"], as_index=False).size().rename(columns={"size": "Employees"})
    st.plotly_chart(px.bar(ot, x="OverTime", y="Employees", color="Attrition", barmode="group",
                           title="Attrition split by overtime"), use_container_width=True)

st.divider()
st.subheader("Employee attrition prediction (demonstration)")
st.write("A Random Forest model is trained on the included **synthetic sample data**. Results are for learning and portfolio demonstration only—not for real HR decisions.")
features = ["Age", "Department", "Education", "JobLevel", "YearsExperience", "YearsAtCompany",
            "MonthlyIncome", "JobSatisfaction", "WorkLifeBalance", "PerformanceRating",
            "OverTime", "YearsSincePromotion"]
X = df[features]
y = df["Attrition"].map({"No": 0, "Yes": 1})
numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [col for col in features if col not in numeric]
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
model = Pipeline([("prep", preprocessor), ("classifier", RandomForestClassifier(
    n_estimators=180, max_depth=8, class_weight="balanced", random_state=42))])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy_score(y_test, pred):.1%}")
m2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.1%}")
m3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.1%}")
m4.metric("F1 score", f"{f1_score(y_test, pred, zero_division=0):.1%}")

with st.expander("Try a sample employee prediction"):
    p1, p2, p3 = st.columns(3)
    with p1:
        p_age = st.slider("Age", 18, 65, 30, key="p_age")
        p_dept = st.selectbox("Department", sorted(df["Department"].unique()), key="p_dept")
        p_edu = st.selectbox("Education", sorted(df["Education"].unique()), key="p_edu")
        p_level = st.selectbox("Job level", sorted(df["JobLevel"].unique()), key="p_level")
    with p2:
        p_exp = st.slider("Years of experience", 0, 40, 6, key="p_exp")
        p_company = st.slider("Years at company", 0, 30, 2, key="p_company")
        p_income = st.number_input("Monthly income (₹)", 1000, 50000, 3500, step=500, key="p_income")
        p_sat = st.slider("Job satisfaction (1–5)", 1.0, 5.0, 3.0, step=0.5, key="p_sat")
    with p3:
        p_wlb = st.slider("Work-life balance (1–5)", 1.0, 5.0, 3.0, step=0.5, key="p_wlb")
        p_perf = st.slider("Performance rating (1–5)", 1.0, 5.0, 3.0, step=0.5, key="p_perf")
        p_ot = st.selectbox("Overtime", ["No", "Yes"], key="p_ot")
        p_promo = st.slider("Years since promotion", 0, 20, 2, key="p_promo")
    if st.button("Predict attrition likelihood"):
        sample = pd.DataFrame([{
            "Age": p_age, "Department": p_dept, "Education": p_edu, "JobLevel": p_level,
            "YearsExperience": p_exp, "YearsAtCompany": min(p_company, p_exp),
            "MonthlyIncome": p_income, "JobSatisfaction": p_sat, "WorkLifeBalance": p_wlb,
            "PerformanceRating": p_perf, "OverTime": p_ot, "YearsSincePromotion": p_promo}])
        probability = model.predict_proba(sample)[0][1]
        st.metric("Estimated attrition probability", f"{probability:.1%}")
        st.caption("This is a model estimate from synthetic data, not a reliable judgment about a real person.")

st.divider()
st.subheader("Explore employee records")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                   file_name="filtered_hr_employee_data.csv", mime="text/csv")
st.caption("Portfolio project by Nitesh Kumar | Built with Python, Pandas, Streamlit, Plotly, and scikit-learn.")
