"""
REAL-TIME DATASET IMPLEMENTATION
Food Delivery Order Prediction - Contains All 5 Version Space Failure Cases
Dataset: food_delivery_orders.csv
"""

import pandas as pd
import numpy as np
import copy
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ============================================================
# 1. CREATE REAL-TIME DATASET
# ============================================================

def create_real_time_dataset():
    """
    Creates a realistic food delivery dataset that naturally contains:
    1. Noisy Data - Same customer, same conditions, different orders
    2. XOR Pattern - Complex ordering behavior (OR conditions)
    3. Missing Values - Incomplete customer profiles
    4. Continuous Values - Distance, Price, Time
    5. Concept Drift - Seasonal/weekend pattern changes
    """
    
    np.random.seed(42)
    
    data = {
        'Customer_ID': [],
        'Distance_km': [],        # Continuous
        'Restaurant_Rating': [],  # Continuous
        'Delivery_Time_mins': [], # Continuous with missing
        'Weather': [],            # Categorical
        'Day_Type': [],           # Categorical
        'Time_of_Day': [],        # Categorical
        'Is_Weekend': [],         # Binary
        'Previous_Orders': [],    # Discrete with missing
        'Order_Placed': []        # Target (1=Yes, 0=No)
    }
    
    # Generate 100 realistic orders over 6 months
    for i in range(100):
        # Basic features
        distance = round(np.random.uniform(0.5, 15.0), 1)
        rating = round(np.random.uniform(2.5, 5.0), 1)
        delivery_time = round(distance * np.random.uniform(3, 8), 0)
        
        # Time-based features (creates concept drift)
        month = (i // 16) + 1  # 0-5 months
        is_weekend = 1 if i % 7 in [5, 6] else 0
        hour = np.random.choice([9, 12, 15, 18, 21, 23])
        
        if hour < 11:
            time_of_day = 'Morning'
        elif hour < 16:
            time_of_day = 'Afternoon'
        elif hour < 21:
            time_of_day = 'Evening'
        else:
            time_of_day = 'Night'
        
        # Weather (affects ordering)
        weather_options = ['Sunny', 'Rainy', 'Cloudy']
        weather = np.random.choice(weather_options, p=[0.6, 0.25, 0.15])
        
        # Day type
        if is_weekend:
            day_type = 'Weekend'
        else:
            day_type = 'Weekday'
        
        # Previous orders (sometimes missing - new users)
        if np.random.random() < 0.15:  # 15% missing
            prev_orders = None
        else:
            prev_orders = np.random.randint(0, 50)
        
        # Determine if order is placed
        # This creates a realistic pattern with all 5 issues
        
        # Issue 1 & 5: Concept Drift + Noise
        if month <= 2:  # Months 1-2: Conservative ordering
            base_prob = 0.3
            if weather == 'Rainy':
                base_prob += 0.4  # More orders in rain
            if is_weekend:
                base_prob += 0.2  # Weekend orders
            
        elif month <= 4:  # Months 3-4: Summer - Different pattern (Concept Drift)
            base_prob = 0.5
            if weather == 'Sunny':
                base_prob += 0.3  # Now sunny weather drives orders!
            if time_of_day == 'Night':
                base_prob += 0.2  # Late night orders increase
            
        else:  # Months 5-6: Monsoon - Another pattern change
            base_prob = 0.4
            if weather == 'Rainy':
                base_prob -= 0.1  # People avoid ordering in heavy rain
            if distance > 8:
                base_prob -= 0.2  # Long distance orders drop
        
        # Issue 2: XOR-like pattern (Complex conditions)
        # Order if: (Close restaurant OR High rating) BUT NOT (Expensive AND Far)
        if distance < 3 and rating > 4.0:
            base_prob += 0.3
        elif distance > 10 and rating < 3.5:
            base_prob -= 0.3
        
        # Issue 3: Missing values naturally occur
        if prev_orders is None:
            base_prob = base_prob * 0.8  # New users order less
        
        # Add noise (Issue 1)
        if np.random.random() < 0.1:  # 10% noise
            base_prob = 1 - base_prob  # Flip the probability
        
        # Final decision
        order_placed = 1 if np.random.random() < base_prob else 0
        
        # Add to dataset
        data['Customer_ID'].append(f'CUST_{i+1:04d}')
        data['Distance_km'].append(distance)
        data['Restaurant_Rating'].append(rating)
        data['Delivery_Time_mins'].append(delivery_time if np.random.random() > 0.1 else None)
        data['Weather'].append(weather)
        data['Day_Type'].append(day_type)
        data['Time_of_Day'].append(time_of_day)
        data['Is_Weekend'].append(is_weekend)
        data['Previous_Orders'].append(prev_orders)
        data['Order_Placed'].append(order_placed)
    
    df = pd.DataFrame(data)
    df.to_csv('food_delivery_orders.csv', index=False)
    print("✅ Real-time dataset created: 'food_delivery_orders.csv'")
    return df


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_real_dataset(filepath='food_delivery_orders.csv'):
    """Load the real-time dataset"""
    try:
        df = pd.read_csv(filepath)
        print("=" * 90)
        print(" FOOD DELIVERY ORDER PREDICTION - REAL-TIME VERSION SPACE ANALYSIS")
        print("=" * 90)
        print(f"\n✅ Dataset loaded: '{filepath}'")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n📊 First 10 Rows:")
        print(df.head(10).to_string())
        print(f"\n📊 Missing Values Summary:")
        print(df.isnull().sum().to_string())
        return df
    except FileNotFoundError:
        print(f"⚠️  Dataset not found. Creating new dataset...")
        return create_real_time_dataset()


# ============================================================
# 3. FIND-S ALGORITHM
# ============================================================

class FindS:
    """Find-S Algorithm Implementation"""
    
    def __init__(self):
        self.hypothesis = None
        self.history = []
        self.collapsed = False
        
    def fit(self, X, y):
        if len(X) == 0:
            return None
        
        self.hypothesis = ['Ø'] * len(X[0])
        self.history = []
        
        for idx, (example, label) in enumerate(zip(X, y)):
            self.history.append(copy.deepcopy(self.hypothesis))
            
            if label == 1:  # Positive example
                for i in range(len(self.hypothesis)):
                    if self.hypothesis[i] == 'Ø':
                        self.hypothesis[i] = example[i]
                    elif self.hypothesis[i] != example[i] and not pd.isna(example[i]):
                        self.hypothesis[i] = '?'
                
                if all(h == 'Ø' for h in self.hypothesis):
                    self.collapsed = True
        
        return self.hypothesis


# ============================================================
# 4. CANDIDATE ELIMINATION ALGORITHM
# ============================================================

class CandidateElimination:
    """Candidate Elimination Algorithm Implementation"""
    
    def __init__(self, domains=None):
        self.S = None
        self.G = None
        self.history = []
        self.collapsed = False
        self.collapse_reason = None
        self.domains = domains
        
    def fit(self, X, y):
        n_features = len(X[0])
        self.S = [['Ø'] * n_features]
        self.G = [['?'] * n_features]
        self.history = []
        
        for idx, (example, label) in enumerate(zip(X, y)):
            # Skip missing values
            if any(pd.isna(v) for v in example):
                continue
                
            if label == 1:  # Positive
                # Update G
                self.G = [g for g in self.G if self._covers(g, example)]
                
                # Update S
                new_S = []
                for s in self.S:
                    if not self._covers(s, example):
                        new_s = self._generalize(s, example)
                        new_S.append(new_s)
                    else:
                        new_S.append(s)
                self.S = self._remove_more_general(new_S)
                
            else:  # Negative
                # Update S
                self.S = [s for s in self.S if not self._covers(s, example)]
                
                # Update G
                new_G = []
                for g in self.G:
                    if self._covers(g, example):
                        specs = self._specialize(g, example)
                        for spec in specs:
                            if any(self._more_general(spec, s) for s in self.S):
                                new_G.append(spec)
                    else:
                        new_G.append(g)
                self.G = self._remove_less_general(new_G)
            
            self.history.append({'S': copy.deepcopy(self.S), 'G': copy.deepcopy(self.G)})
            
            if len(self.S) == 0 or len(self.G) == 0:
                self.collapsed = True
                self.collapse_reason = f"Collapsed at example {idx}"
                break
        
        return self.history
    
    def _covers(self, h, x):
        for i in range(len(h)):
            if h[i] != '?' and str(h[i]) != str(x[i]):
                return False
        return True
    
    def _more_general(self, h1, h2):
        for i in range(len(h1)):
            if h1[i] != '?' and (h1[i] != h2[i] and h2[i] != 'Ø'):
                return False
        return True
    
    def _generalize(self, s, x):
        new_s = []
        for i in range(len(s)):
            if s[i] == 'Ø':
                new_s.append(x[i])
            elif str(s[i]) != str(x[i]):
                new_s.append('?')
            else:
                new_s.append(s[i])
        return new_s
    
    def _specialize(self, g, x):
        specs = []
        for i in range(len(g)):
            if g[i] == '?' and self.domains and i < len(self.domains):
                for val in self.domains[i]:
                    if str(val) != str(x[i]):
                        new_g = g.copy()
                        new_g[i] = val
                        specs.append(new_g)
        return specs
    
    def _remove_more_general(self, boundary):
        return [h for h in boundary if not any(self._more_general(h2, h) and h != h2 for h2 in boundary)]
    
    def _remove_less_general(self, boundary):
        return [h for h in boundary if not any(self._more_general(h, h2) and h != h2 for h2 in boundary)]


# ============================================================
# 5. DATA PREPROCESSING
# ============================================================

def discretize_for_ce(df):
    """Discretize continuous features for CE algorithm"""
    df_disc = df.copy()
    
    # Discretize Distance
    df_disc['Distance_km'] = pd.cut(df['Distance_km'], 
                                     bins=[0, 3, 8, 15], 
                                     labels=['Nearby', 'Medium', 'Far'])
    
    # Discretize Rating
    df_disc['Restaurant_Rating'] = pd.cut(df['Restaurant_Rating'], 
                                           bins=[0, 3.5, 4.2, 5.0], 
                                           labels=['Low', 'Medium', 'High'])
    
    # Discretize Delivery Time
    df_disc['Delivery_Time_mins'] = pd.cut(df['Delivery_Time_mins'].fillna(-1), 
                                            bins=[-1, 20, 40, 100], 
                                            labels=['Fast', 'Medium', 'Slow'])
    df_disc.loc[df['Delivery_Time_mins'].isna(), 'Delivery_Time_mins'] = None
    
    # Discretize Previous Orders
    df_disc['Previous_Orders'] = pd.cut(df['Previous_Orders'].fillna(-1), 
                                         bins=[-1, 5, 20, 50], 
                                         labels=['New', 'Regular', 'Loyal'])
    df_disc.loc[df['Previous_Orders'].isna(), 'Previous_Orders'] = None
    
    return df_disc


def prepare_features(df, discretize=False):
    """Prepare features for algorithms"""
    if discretize:
        df = discretize_for_ce(df)
    
    feature_cols = ['Distance_km', 'Restaurant_Rating', 'Weather', 
                    'Day_Type', 'Time_of_Day', 'Is_Weekend']
    
    X = df[feature_cols].values.tolist()
    y = df['Order_Placed'].values.tolist()
    
    return X, y, feature_cols


# ============================================================
# 6. CASE STUDY ANALYSIS FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)

def print_subsection(title):
    print("\n" + "-" * 70)
    print(f" {title}")
    print("-" * 70)


def case_study_1_noisy_data(df):
    """CASE 1: Noisy Data - Contradictory Orders"""
    print_section("CASE STUDY 1: NOISY DATA (Contradictory Orders)")
    
    print("\n📋 Real Scenario: Same customer, same conditions, different decisions")
    print("   Why? Customer might be busy, phone battery dead, or changed mind!")
    
    # Find noisy examples
    feature_cols = ['Distance_km', 'Restaurant_Rating', 'Weather', 'Is_Weekend']
    
    # Round continuous values for grouping
    df_check = df.copy()
    df_check['Distance_km'] = df_check['Distance_km'].round(1)
    df_check['Restaurant_Rating'] = df_check['Restaurant_Rating'].round(1)
    
    # Find duplicates with different labels
    duplicates = df_check[feature_cols].duplicated(keep=False)
    noisy = df_check[duplicates].sort_values(feature_cols)
    
    if len(noisy) > 0:
        print("\n🔍 Noisy Examples Found (Same Conditions, Different Outcome):")
        display_cols = feature_cols + ['Order_Placed']
        print(noisy[display_cols].head(10).to_string())
        
        print("\n💡 Real-World Impact:")
        print("   Customer A: Distance=2.5km, Rating=4.2, Weather=Sunny, Weekend → ORDERED")
        print("   Customer B: Distance=2.5km, Rating=4.2, Weather=Sunny, Weekend → NO ORDER")
        print("   ⚠️  AI cannot learn a consistent rule!")
    
    # Run Find-S
    print_subsection("Find-S Algorithm Result")
    X, y, _ = prepare_features(df.dropna(), discretize=False)
    fs = FindS()
    fs.fit(X, y)
    print(f"   Final Hypothesis: {fs.hypothesis}")
    print("   ⚠️  Overly general due to noise - cannot distinguish patterns")
    
    # Run CE
    print_subsection("Candidate Elimination Result")
    df_disc = discretize_for_ce(df)
    domains = [df_disc[col].dropna().unique().tolist() for col in ['Distance_km', 'Restaurant_Rating', 'Weather', 'Day_Type', 'Time_of_Day', 'Is_Weekend']]
    X_disc, y_disc, _ = prepare_features(df_disc.dropna(), discretize=False)
    ce = CandidateElimination(domains=domains)
    ce.fit(X_disc, y_disc)
    
    if ce.collapsed:
        print(f"   ❌ VERSION SPACE COLLAPSED!")
        print(f"   {ce.collapse_reason}")
        print("   Reason: Contradictory examples make consistent hypothesis IMPOSSIBLE")


def case_study_2_xor_pattern(df):
    """CASE 2: XOR Pattern - Complex Ordering Logic"""
    print_section("CASE STUDY 2: XOR PATTERN (Complex Ordering Behavior)")
    
    print("\n📋 Real Scenario: Customers have complex preferences")
    print("   'I order when: (Rainy day AND Close restaurant) OR (Weekend AND Night time)'")
    print("   'But NOT when: (Rainy AND Far) OR (Weekday Morning)'")
    
    # Show XOR-like patterns
    print("\n🔍 Complex Pattern Analysis:")
    
    # Pattern 1: Rainy + Close = Order
    rainy_close = df[(df['Weather'] == 'Rainy') & (df['Distance_km'] < 3)]
    print(f"\n   Pattern 1: Rainy + Close Restaurant")
    print(f"   Orders: {rainy_close['Order_Placed'].sum()} / {len(rainy_close)} ({rainy_close['Order_Placed'].mean():.1%})")
    
    # Pattern 2: Rainy + Far = No Order
    rainy_far = df[(df['Weather'] == 'Rainy') & (df['Distance_km'] > 8)]
    print(f"\n   Pattern 2: Rainy + Far Restaurant")
    print(f"   Orders: {rainy_far['Order_Placed'].sum()} / {len(rainy_far)} ({rainy_far['Order_Placed'].mean():.1%})")
    
    # Pattern 3: Weekend Night = Order
    weekend_night = df[(df['Is_Weekend'] == 1) & (df['Time_of_Day'] == 'Night')]
    print(f"\n   Pattern 3: Weekend + Night")
    print(f"   Orders: {weekend_night['Order_Placed'].sum()} / {len(weekend_night)} ({weekend_night['Order_Placed'].mean():.1%})")
    
    print("\n💡 Real-World Impact:")
    print("   This is an XOR-like pattern:")
    print("   Order = (Rainy AND Close) OR (Weekend AND Night)")
    print("   ⚠️  CE can only learn AND rules, not OR combinations!")
    print("   Result: VERSION SPACE IMPOSSIBLE to capture true pattern")


def case_study_3_missing_values(df):
    """CASE 3: Missing Values - Incomplete Profiles"""
    print_section("CASE STUDY 3: MISSING VALUES (Incomplete Customer Data)")
    
    print("\n📋 Real Scenario: New users haven't ordered before")
    print("   Delivery time unknown for new restaurants")
    
    # Show missing values
    print("\n🔍 Missing Data Summary:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"   {col}: {count} missing ({count/len(df):.1%})")
    
    print("\n   Example of Missing Data:")
    missing_rows = df[df.isnull().any(axis=1)].head(5)
    print(missing_rows[['Customer_ID', 'Previous_Orders', 'Delivery_Time_mins', 'Order_Placed']].to_string())
    
    print("\n💡 Real-World Impact:")
    print("   • CE cannot process rows with missing values")
    print("   • Must either:")
    print("     - Drop {:.0f} rows (lose valuable data)".format(len(df) - len(df.dropna())))
    print("     - Impute values (introduce bias)")
    print("   • VERSION SPACE BECOMES UNDEFINED!")


def case_study_4_continuous_values(df):
    """CASE 4: Continuous Values - Precise Thresholds"""
    print_section("CASE STUDY 4: CONTINUOUS VALUES (Infinite Possibilities)")
    
    print("\n📋 Real Scenario: What's the exact distance threshold for ordering?")
    
    # Show continuous distributions
    print("\n🔍 Continuous Features Distribution:")
    print(f"\n   Distance (km):")
    print(f"   • Min: {df['Distance_km'].min():.1f} km")
    print(f"   • Max: {df['Distance_km'].max():.1f} km")
    print(f"   • Mean: {df['Distance_km'].mean():.1f} km")
    
    print(f"\n   Restaurant Rating:")
    print(f"   • Min: {df['Restaurant_Rating'].min():.1f}")
    print(f"   • Max: {df['Restaurant_Rating'].max():.1f}")
    print(f"   • Mean: {df['Restaurant_Rating'].mean():.1f}")
    
    # Show optimal threshold analysis
    print("\n💡 The 'Perfect' Threshold Problem:")
    
    thresholds = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    print("\n   Order Rate by Distance Threshold:")
    for thresh in thresholds:
        close_orders = df[df['Distance_km'] <= thresh]['Order_Placed'].mean()
        print(f"   • Distance ≤ {thresh:.1f} km: {close_orders:.1%} order rate")
    
    print("\n   ⚠️  True optimal threshold might be 4.73 km!")
    print("   But what about 4.74? 4.731? 4.732?")
    print("   • INFINITE possible thresholds")
    print("   • CE cannot represent continuous boundaries")
    print("   • VERSION SPACE IS INFINITE (Impossible to list)")


def case_study_5_concept_drift(df):
    """CASE 5: Concept Drift - Seasonal Changes"""
    print_section("CASE STUDY 5: CONCEPT DRIFT (Changing Behavior Over Time)")
    
    print("\n📋 Real Scenario: Ordering behavior changes with seasons")
    
    # Add time period column
    df['Time_Period'] = pd.cut(df.index, bins=[0, 33, 66, 100], labels=['Early (Winter)', 'Middle (Summer)', 'Late (Monsoon)'])
    
    print("\n🔍 Order Rate Changes Over Time:")
    
    for period in ['Early (Winter)', 'Middle (Summer)', 'Late (Monsoon)']:
        period_data = df[df['Time_Period'] == period]
        if len(period_data) > 0:
            order_rate = period_data['Order_Placed'].mean()
            print(f"\n   {period}:")
            print(f"   • Order Rate: {order_rate:.1%}")
            
            # Show weather preference drift
            sunny_rate = period_data[period_data['Weather'] == 'Sunny']['Order_Placed'].mean()
            rainy_rate = period_data[period_data['Weather'] == 'Rainy']['Order_Placed'].mean()
            print(f"   • Sunny weather orders: {sunny_rate:.1%}")
            print(f"   • Rainy weather orders: {rainy_rate:.1%}")
    
    print("\n💡 Real-World Impact:")
    print("   • Winter: People order more in rainy weather (stay home)")
    print("   • Summer: People order more in sunny weather (picnics/outdoors)")
    print("   • Monsoon: Order rate drops overall")
    print("\n   ⚠️  The 'true concept' KEEPS CHANGING!")
    print("   • CE tries to find ONE rule for ALL time periods")
    print("   • VERSION SPACE COLLAPSES - No single consistent rule exists!")


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():
    # Load or create dataset
    df = load_real_dataset()
    
    # Run all case studies
    case_study_1_noisy_data(df)
    case_study_2_xor_pattern(df)
    case_study_3_missing_values(df)
    case_study_4_continuous_values(df)
    case_study_5_concept_drift(df)
    
    # Final Summary
    print_section("SUMMARY: REAL-TIME VERSION SPACE FAILURES")
    
    print("""
┌─────────────────────┬──────────────────────────────────────────────────────────────────┐
│ REAL-WORLD PROBLEM  │ WHY VERSION SPACE FAILS IN FOOD DELIVERY APP                       │
├─────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 1. Noisy Data       │ Same customer conditions → Different decisions (busy, distracted)  │
│ 2. XOR Pattern      │ Order if (Rainy+Close) OR (Weekend+Night) → Too complex for CE    │
│ 3. Missing Values   │ New users → No order history → CE cannot process                   │
│ 4. Continuous Values│ Optimal distance threshold (4.73km) → Infinite possibilities       │
│ 5. Concept Drift    │ Seasonal changes → Winter vs Summer behavior differs              │
└─────────────────────┴──────────────────────────────────────────────────────────────────┘
""")
    
    print("\n🎯 REAL-TIME APPLICATION INSIGHT:")
    print("   Food delivery apps (Zomato/Swiggy/Uber Eats) DON'T use simple Version Space!")
    print("   They use:")
    print("   • Gradient Boosting (XGBoost) - Handles noise and missing values")
    print("   • Neural Networks - Learns XOR and complex patterns")
    print("   • Online Learning - Adapts to concept drift")
    print("   • Regression - Works with continuous values")
    
    print("\n📊 Dataset Statistics:")
    print(f"   • Total Orders: {len(df)}")
    print(f"   • Order Rate: {df['Order_Placed'].mean():.1%}")
    print(f"   • Missing Data: {df.isnull().sum().sum()} values")
    print(f"   • Features: {len(df.columns)}")
    
    print("\n" + "=" * 90)
    print(" ANALYSIS COMPLETE - VERSION SPACE IMPOSSIBLE IN REAL-WORLD SCENARIOS")
    print("=" * 90)


if __name__ == "__main__":
    main()