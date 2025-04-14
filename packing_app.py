import streamlit as st
import pandas as pd
import pulp

# Title
st.title("Room Packing Optimizer (IP & DP)")

# Upload item file
uploaded_file = st.file_uploader("Upload Item Excel File", type=['xlsx'])

# Get user input for capacities and costs
st.sidebar.header("Baggage Constraints")
W_c = st.sidebar.number_input("Cabin Weight Limit (kg)", value=15.0)
Vol_c = st.sidebar.number_input("Cabin Volume Limit (liters)", value=44.0)
W_ck = st.sidebar.number_input("Check-in Weight Limit (kg)", value=25.0)
Vol_ck = st.sidebar.number_input("Check-in Volume Limit (liters)", value=116.13)

st.sidebar.header("Movers Cost & Penalty")
C_pm = st.sidebar.number_input("Cost per liter for Packers & Movers (₹)", value=1.0)
M = st.sidebar.number_input("Penalty for unsuitable items to Movers (₹)", value=100000.0)

if uploaded_file:
    data = pd.read_excel(uploaded_file)
    
    # Value adjustment
    data['Current Monetary Value'] = data['Monetary Value (INR)'] * ((1 - 0.01 * data['Depreciation per Year (%)']) ** data['Age (years)'])
    data.drop(columns=['Monetary Value (INR)', 'Depreciation per Year (%)', 'Age (years)'], inplace=True)
    
    data['Ac'] = data['Airline Baggage Type'].apply(lambda x: 1 if 'Hand Baggage' in x else 0)
    data['Ack'] = data['Airline Baggage Type'].apply(lambda x: 1 if 'Check-in Baggage' in x else 0)
    data['Apm'] = 1  # Assume all items can go to movers

    items = data['Item'].tolist()
    n = len(items)
    v = data['Current Monetary Value'].tolist()
    w = data['Weight (kg)'].tolist()
    u = data['Volume (liters)'].tolist()
    A_c = data['Ac'].tolist()
    A_ck = data['Ack'].tolist()
    A_pm = data['Apm'].tolist()

    st.subheader("Solving with Integer Programming...")

    # IP Model
    prob = pulp.LpProblem("Packing_Optimization", pulp.LpMaximize)
    x_c = pulp.LpVariable.dicts("Cabin", items, cat='Binary')
    x_ck = pulp.LpVariable.dicts("Checkin", items, cat='Binary')
    x_pm = pulp.LpVariable.dicts("Movers", items, cat='Binary')

    prob += (
        pulp.lpSum(v[i] * x_c[items[i]] for i in range(n)) +
        pulp.lpSum(v[i] * x_ck[items[i]] for i in range(n)) -
        C_pm * pulp.lpSum(u[i] * x_pm[items[i]] for i in range(n)) -
        M * pulp.lpSum((1 - A_pm[i]) * x_pm[items[i]] for i in range(n))
    )

    for i in range(n):
        prob += x_c[items[i]] + x_ck[items[i]] + x_pm[items[i]] <= 1
        prob += x_c[items[i]] <= A_c[i]
        prob += x_ck[items[i]] <= A_ck[i]
        prob += x_pm[items[i]] <= A_pm[i]

    prob += pulp.lpSum(w[i] * x_c[items[i]] for i in range(n)) <= W_c
    prob += pulp.lpSum(u[i] * x_c[items[i]] for i in range(n)) <= Vol_c
    prob += pulp.lpSum(w[i] * x_ck[items[i]] for i in range(n)) <= W_ck
    prob += pulp.lpSum(u[i] * x_ck[items[i]] for i in range(n)) <= Vol_ck

    prob.solve()

    cabin_items_ip = [i for i in items if x_c[i].varValue == 1]
    checkin_items_ip = [i for i in items if x_ck[i].varValue == 1]
    movers_items_ip = [i for i in items if x_pm[i].varValue == 1]

    # DP Solver
    st.subheader("Solving with Dynamic Programming...")

    data_sorted = data.sort_values(by='Current Monetary Value', ascending=False).reset_index(drop=True)

    def dp_solver(data_sorted):
        items = data_sorted['Item'].tolist()
        values = data_sorted['Current Monetary Value'].tolist()
        weights = data_sorted['Weight (kg)'].tolist()
        volumes = data_sorted['Volume (liters)'].tolist()
        Ac = data_sorted['Ac'].tolist()
        Ack = data_sorted['Ack'].tolist()
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
        cabin_items = []
        w, v = int(W_c), int(Vol_c)
        for i in range(n, 0, -1):
            if dp_c[i][w][v] != dp_c[i-1][w][v]:
                cabin_items.append(items[i-1])
                w -= int(weights[i-1])
                v -= int(volumes[i-1])

        # Check-in DP
        selected_cabin = set(cabin_items)
        dp_ck = [[[0]*(int(Vol_ck)+1) for _ in range(int(W_ck)+1)] for _ in range(n+1)]
        for i in range(1, n+1):
            for w in range(int(W_ck)+1):
                for v in range(int(Vol_ck)+1):
                    dp_ck[i][w][v] = dp_ck[i-1][w][v]
                    if items[i-1] not in selected_cabin and w >= weights[i-1] and v >= volumes[i-1] and Ack[i-1] == 1:
                        dp_ck[i][w][v] = max(dp_ck[i][w][v],
                                             dp_ck[i-1][int(w-weights[i-1])][int(v-volumes[i-1])] + values[i-1])
        checkin_items = []
        w, v = int(W_ck), int(Vol_ck)
        for i in range(n, 0, -1):
            if items[i-1] not in selected_cabin and dp_ck[i][w][v] != dp_ck[i-1][w][v]:
                checkin_items.append(items[i-1])
                w -= int(weights[i-1])
                v -= int(volumes[i-1])

        movers_items = [i for i in items if i not in cabin_items and i not in checkin_items]
        return cabin_items, checkin_items, movers_items

    cabin_items_dp, checkin_items_dp, movers_items_dp = dp_solver(data_sorted)

    # Display Results
    st.markdown("### 🧠 Integer Programming Results")
    st.write("**Cabin Items:**", cabin_items_ip)
    st.write("**Check-in Items:**", checkin_items_ip)
    st.write("**Movers Items:**", movers_items_ip)

    st.markdown("### ⚙️ Dynamic Programming Results")
    st.write("**Cabin Items:**", cabin_items_dp)
    st.write("**Check-in Items:**", checkin_items_dp)
    st.write("**Movers Items:**", movers_items_dp)
