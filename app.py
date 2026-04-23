import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config(page_title="House Price Prediction Tool", layout="wide")
st.title("🏡 House Price Prediction Tool")

# Set plot style
plt.rcParams["font.family"] = "DejaVu Sans"
plt.style.use("seaborn-v0_8-whitegrid")

@st.cache_data
def load_and_preprocess_data():
    # Load dataset with absolute path
    file_path = r"D:\HuaweiMoveData\Users\86133\Desktop\train.csv"
    df = pd.read_csv(file_path, encoding="latin-1")

    # Simple data cleaning
    df = df.dropna(axis=1, thresh=len(df)*0.7)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df

@st.cache_resource
def train_model(df):
    # Prepare features and target
    X = df.drop(["SalePrice", "Id"], axis=1)
    y = df["SalePrice"]
    X = pd.get_dummies(X)

    # Train test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Train linear regression for fit line
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_val)
    lr_pred = lr.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    return model, lr, X.columns, X_val, y_val, y_pred, lr_pred, mae, r2

# Load data and model
df = load_and_preprocess_data()
model, lr, feature_cols, X_val, y_val, y_pred, lr_pred, mae, r2 = train_model(df)

# Sidebar user input
st.sidebar.header("House Features")
overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
living_area = st.sidebar.slider("Living Area (sq ft)", 300, 6000, 1500)
garage_area = st.sidebar.slider("Garage Area (sq ft)", 0, 1500, 500)
basement = st.sidebar.slider("Basement Area (sq ft)", 0, 3000, 1000)
year_built = st.sidebar.slider("Year Built", 1800, 2024, 2000)

# Create input DataFrame
input_df = pd.DataFrame({
    "OverallQual": [overall_qual],
    "GrLivArea": [living_area],
    "GarageArea": [garage_area],
    "TotalBsmtSF": [basement],
    "YearBuilt": [year_built]
})

# Display user input
st.subheader("User Input Parameters")
st.dataframe(input_df, hide_index=True)

# Predict button
if st.button("Predict House Price"):
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    result = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${result:,.2f}")

# ----------------------
# Dataset Visualization Section
# ----------------------
st.markdown("---")
st.subheader("📊 Dataset Visualization")

# Row 1
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("SalePrice Distribution")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(df["SalePrice"], bins=25, color="#4895ef", edgecolor="white")
    ax1.set_xlabel("Sale Price")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with col_b:
    st.subheader("Overall Quality vs Price")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    data_to_plot = [df[df["OverallQual"] == i]["SalePrice"] for i in range(1, 11)]
    ax2.boxplot(data_to_plot, labels=range(1, 11))
    ax2.set_xlabel("Overall Quality")
    ax2.set_ylabel("Sale Price")
    st.pyplot(fig2)

# Row 2
col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Living Area vs Price")
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5, color="#f72585")
    ax3.set_xlabel("Living Area (sq ft)")
    ax3.set_ylabel("Sale Price")
    st.pyplot(fig3)

with col_d:
    st.subheader("Year Built vs Average Price")
    year_avg = df.groupby("YearBuilt")["SalePrice"].mean()
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    ax4.plot(year_avg.index, year_avg.values, color="#27ae60", linewidth=1.5)
    ax4.set_xlabel("Year Built")
    ax4.set_ylabel("Avg Sale Price")
    st.pyplot(fig4)

# Linear Regression Fit Line (GrLivArea vs SalePrice)
st.subheader("Linear Regression Fit Line (GrLivArea vs SalePrice)")
x = df["GrLivArea"].values.reshape(-1, 1)
y = df["SalePrice"].values
lr_single = LinearRegression()
lr_single.fit(x, y)
x_pred = np.linspace(x.min(), x.max(), 100)
y_pred_single = lr_single.predict(x_pred.reshape(-1, 1))

fig_lr, ax_lr = plt.subplots(figsize=(10, 5))
ax_lr.scatter(x, y, alpha=0.3, color="#f72585")
ax_lr.plot(x_pred, y_pred_single, color="red", linewidth=2)
ax_lr.set_xlabel("Living Area (sq ft)")
ax_lr.set_ylabel("Sale Price")
ax_lr.set_title("Linear Regression: GrLivArea vs SalePrice")
st.pyplot(fig_lr)

# Residual Plot
st.subheader("Residual Plot (Linear Regression)")
residuals = y - lr_single.predict(x)
fig_res, ax_res = plt.subplots(figsize=(10, 5))
ax_res.scatter(lr_single.predict(x), residuals, alpha=0.6, color="#5f9bf7")
ax_res.axhline(y=0, color="red", linestyle="--")
ax_res.set_xlabel("Predicted Price")
ax_res.set_ylabel("Residual (Actual - Predicted)")
st.pyplot(fig_res)

# All Numeric Features Distribution
st.subheader("All Numeric Features Distribution")
df_num = df.select_dtypes(include=[np.number])
top_n_cols = df_num.columns[:12]
fig_hist, ax_hist = plt.subplots(figsize=(16, 12))
df_num[top_n_cols].hist(bins=30, ax=ax_hist, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
st.pyplot(fig_hist)

# ----------------------
# Model Performance
# ----------------------
st.markdown("---")
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("MAE", f"${mae:,.0f}")
col2.metric("R² Score", f"{r2:.3f}")

# Actual vs Predicted Plot (Random Forest)
st.subheader("Actual vs Predicted Price (Random Forest)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_val, y_pred, alpha=0.6)
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
st.pyplot(fig)

# Feature Importance
st.subheader("Top 10 Feature Importance")
importance = pd.Series(model.feature_importances_, index=feature_cols)
top10 = importance.sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(8, 5))
top10.plot(kind="barh", ax=ax2, color="#57cc99")
st.pyplot(fig2)