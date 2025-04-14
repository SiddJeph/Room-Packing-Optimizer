import streamlit as st
import pandas as pd
import pulp

st.set_page_config(page_title="Room Packing Optimizer", layout="wide")
st.title("🧳 Room Packing Optimizer using IP & DP")

# Sidebar inputs for constraints and costs
st.sidebar.header("Baggage Constraints")
W_c = st.sidebar.number_input("Cabin Weight Limit (kg)", value=15.0)
Vol_c = st.sidebar.number_input("Cabin Volume Limit (liters)", value=44.0)
W_ck = st.sidebar.number_input("Check-in Weight Limit (kg)", value=25.0)
Vol_ck = st.sidebar.number_input("Check-in Volume Limit (liters)", value=116.13)

st.sidebar.header("Packers and Movers Parameters")
C_pm = st.sidebar.number_input("Cost per Liter (₹)", value=1.0)
M = st.sidebar.number_input("Penalty for Incompatible Items (₹)", value=100000.0)

# Session state initialization
if "items" not in st.session_state or not isinstance(st.session_state.items, list):
    st.session_state.items = []

# Item input form
st.subheader("➕ Add Item Details")

with st.form("item_form", clear_on_submit=True):
    name = st.text_input("Item Name")
    value = st.number_input("Current Monetary Value (₹)", min_value=0.0)
    weight = st.number_input("Weight (kg)", min_value=0.0)
    volume = st.number_input("Volume (liters)", min_value=0.0)
    ac = st.checkbox("Cabin Compatible")
    ack = st.checkbox("Check-in Compatible")
    apm = st.checkbox("Movers Compatible", value=True)
    submitted = st.form_submit_button("Add Item")

if submitted and name:
    st.session_state.items.append({
        "Item": name,
        "Value": value,
        "Weight": weight,
        "Volume": volume,
        "Ac": int(ac),
        "Ack": int(ack),
        "Apm": int(apm)
    })

# Show item table
if len(st.session_state.items) > 0:
    data = pd.DataFrame(st.session_state.items)
    st.markdown("### 📦 Item List")
    st.dataframe(data, use_container_width=True)

    # Extract data for optimization
    items = data['Item'].tolist()
    v = data['Value'].tolist()
    w = data['Weight'].tolist()
    u = data['Volume'].tolist()
    A_c = data['Ac'].tolist()
    A_ck = data['Ack'].tolist()
    A_pm = data['Apm'].tolist()
    n = len(items)

    # ========== Integer Programming ==========
    st.markdown("## 🧠 Integer Programming Solution")

    prob = pulp.LpProblem("Packing_Optimization", pulp.LpMaximize)
    x_c = pulp.LpVariable.dicts("Cabin", items, cat='Binary')
    x_ck = pulp.LpVariable.dicts("Checkin", items, cat='Binary')
    x_pm = pulp.LpVariable.dicts("Movers", items, cat='Binary')

    # Objective
    prob += (
        pulp.lpSum(v[i] * x_c[items[i]] for i in range(n)) +
        pulp.lpSum(v[i] * x_ck[items[i]] for i in range(n)) -
        C_pm * pulp.lpSum(u[i] * x_pm[items[i]] for i in range(n)) -
        M * pulp.lpSum((1 - A_pm[i]) * x_pm[items[i]] for i in range(n))
    )

    # Constraints
    for i in range(n):
        prob += x_c[items[i]] + x_ck[items[i]] + x_pm[items[i]] <= 1
        prob += x_c[items[i]] <= A_c[i]
        prob += x_ck[items[i]] <= A_ck[i]
        prob += x_pm[items[i]] <= A_pm[i]

    prob += pulp.lpSum(w[i] * x_c[items[i]] for i in range(n)) <= W_c
    prob += pulp.lpSum(u[i] * x_c[items[i]] for i in range(n)) <= Vol_c
    prob += pulp.lpSum(w[i] * x_ck[items[i]] for i in range(n)) <= W_ck
    prob += pulp.lpSum(u[i] * x_ck[items[i]] for i in range(n)) <= Vol_ck

    # Solve
    prob.solve()

    cabin_ip = [i for i in items if x_c[i].varValue == 1]
    checkin_ip = [i for i in items if x_ck[i].varValue == 1]
    movers_ip = [i for i in items if x_pm[i].varValue == 1]

    st.write("**Cabin Items (IP):**", cabin_ip)
    st.write("**Check-in Items (IP):**", checkin_ip)
    st.write("**Movers Items (IP):**", movers_ip)

    # ========== Dynamic Programming ==========
    st.markdown("## ⚙️ Dynamic Programming Heuristic")

    data_sorted = data.sort_values(by='Value', ascending=False).reset_index(drop=True)
    items = data_sorted['Item'].tolist()
    values = data_sorted['Value'].tolist()
    weights = data_sorted['Weight'].tolist()
    volumes = data_sorted['Volume'].tolist()
    Ac = data_sorted['Ac'].tolist()
    Ack = data_sorted['Ack'].tolist()
    Apm = data_sorted['Apm'].tolist()
    n = len(items)

    # Cabin DP
    dp_c = [[[0]*(int(Vol_c)+1) for _ in range(int(W_c)+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(int(W_c)+1):
            for v in range(int(Vol_c)+1):
                dp_c[i][w][v] = dp_c[i-1][w][v]
                if w >= weights[i-1] and v >= volumes[i-1] and Ac[i-1] == 1:
                    dp_c[i][w][v] = max(dp_c[i][w][v],
                                        dp_c[i-1][int(w-weights[i-1])][int(v-volumes[i-1])] + values[i-1])
    cabin_dp = []
    w, v = int(W_c), int(Vol_c)
    for i in range(n, 0, -1):
        if dp_c[i][w][v] != dp_c[i-1][w][v]:
            cabin_dp.append(items[i-1])
            w -= int(weights[i-1])
            v -= int(volumes[i-1])

    # Check-in DP
    selected_cabin = set(cabin_dp)
    dp_ck = [[[0]*(int(Vol_ck)+1) for _ in range(int(W_ck)+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(int(W_ck)+1):
            for v in range(int(Vol_ck)+1):
                dp_ck[i][w][v] = dp_ck[i-1][w][v]
                if items[i-1] not in selected_cabin and w >= weights[i-1] and v >= volumes[i-1] and Ack[i-1] == 1:
                    dp_ck[i][w][v] = max(dp_ck[i][w][v],
                                         dp_ck[i-1][int(w-weights[i-1])][int(v-volumes[i-1])] + values[i-1])
    checkin_dp = []
    w, v = int(W_ck), int(Vol_ck)
    for i in range(n, 0, -1):
        if items[i-1] not in selected_cabin and dp_ck[i][w][v] != dp_ck[i-1][w][v]:
            checkin_dp.append(items[i-1])
            w -= int(weights[i-1])
            v -= int(volumes[i-1])

    movers_dp = [i for i in items if i not in cabin_dp and i not in checkin_dp]

    st.write("**Cabin Items (DP):**", cabin_dp)
    st.write("**Check-in Items (DP):**", checkin_dp)
    st.write("**Movers Items (DP):**", movers_dp)

else:
    st.info("Add items using the form to start solving.")
