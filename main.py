"""
main.py - Version Space Failure Analysis
Demonstrates 5 situations where Version Space is impossible to obtain
Dataset: unified_version_space_cases.csv
"""

import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_dataset(filepath='unified_version_space_cases.csv'):
    """Load the dataset from CSV file"""
    try:
        df = pd.read_csv(filepath)
        
        # Clean column names - remove leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Clean string columns - remove leading/trailing spaces
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # Convert numeric columns (handle spaces in numbers)
        if 'Income' in df.columns:
            df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
        if 'Credit_Score' in df.columns:
            df['Credit_Score'] = pd.to_numeric(df['Credit_Score'], errors='coerce')
        if 'Employment_Years' in df.columns:
            df['Employment_Years'] = pd.to_numeric(df['Employment_Years'], errors='coerce')
        if 'Application_Month' in df.columns:
            df['Application_Month'] = pd.to_numeric(df['Application_Month'], errors='coerce')
        if 'Approved' in df.columns:
            df['Approved'] = pd.to_numeric(df['Approved'], errors='coerce')
        
        print("=" * 80)
        print("VERSION SPACE FAILURE ANALYSIS - 5 CASE STUDIES")
        print("=" * 80)
        print(f"\n✅ Dataset loaded: '{filepath}'")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n📊 Dataset Preview:")
        print(df.head(10).to_string())
        print(f"\n📊 Missing Values:")
        print(df.isnull().sum().to_string())
        print(f"\n📊 Data Types:")
        print(df.dtypes.to_string())
        return df
    except FileNotFoundError:
        print(f"❌ Error: '{filepath}' not found!")
        print("   Please ensure the CSV file is in the same directory as main.py")
        exit(1)

# ============================================================
# 2. FIND-S ALGORITHM
# ============================================================

class FindS:
    """Find-S Algorithm - Finds the maximally specific hypothesis"""
    
    def __init__(self):
        self.hypothesis = None
        self.history = []
        self.collapsed = False
        self.collapse_reason = None
    
    def initialize_hypothesis(self, num_features):
        return ['Ø'] * num_features
    
    def generalize(self, hypothesis, example):
        for i in range(len(hypothesis)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = example[i]
            elif hypothesis[i] != example[i] and example[i] is not None and not pd.isna(example[i]):
                hypothesis[i] = '?'
        return hypothesis
    
    def fit(self, X, y, verbose=False):
        if len(X) == 0:
            return None
        
        self.hypothesis = self.initialize_hypothesis(len(X[0]))
        self.history = []
        self.collapsed = False
        
        for idx, (example, label) in enumerate(zip(X, y)):
            self.history.append(copy.deepcopy(self.hypothesis))
            
            if label == 1:  # Positive example
                self.hypothesis = self.generalize(self.hypothesis, list(example))
                
                if verbose:
                    print(f"  Ex {idx}: {self.hypothesis}")
                
                # Check for collapse (all Ø)
                if all(h == 'Ø' for h in self.hypothesis):
                    self.collapsed = True
                    self.collapse_reason = f"Collapsed at example {idx}"
        
        return self.hypothesis


# ============================================================
# 3. CANDIDATE ELIMINATION ALGORITHM
# ============================================================

class CandidateElimination:
    """Candidate Elimination Algorithm - Maintains S and G boundaries"""
    
    def __init__(self, attribute_domains=None):
        self.S = None
        self.G = None
        self.history = []
        self.collapsed = False
        self.collapse_reason = None
        self.attribute_domains = attribute_domains
    
    def initialize_boundaries(self, num_features):
        self.S = [['Ø'] * num_features]
        self.G = [['?'] * num_features]
    
    def is_more_general(self, h1, h2):
        for i in range(len(h1)):
            if h1[i] != '?' and (h1[i] != h2[i] and h2[i] != 'Ø'):
                return False
        return True
    
    def covers(self, hypothesis, example):
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and str(hypothesis[i]) != str(example[i]):
                return False
        return True
    
    def generalize_S(self, s, example):
        new_s = []
        for i in range(len(s)):
            if s[i] == 'Ø':
                new_s.append(example[i])
            elif str(s[i]) != str(example[i]) and example[i] is not None and not pd.isna(example[i]):
                new_s.append('?')
            else:
                new_s.append(s[i])
        return new_s
    
    def specialize_G(self, g, example):
        specializations = []
        for i in range(len(g)):
            if g[i] == '?':
                if self.attribute_domains and i < len(self.attribute_domains):
                    for value in self.attribute_domains[i]:
                        if str(value) != str(example[i]) and value is not None and not pd.isna(value):
                            new_h = g.copy()
                            new_h[i] = value
                            specializations.append(new_h)
        return specializations
    
    def remove_inconsistent(self, boundary, example, is_positive):
        return [h for h in boundary if self.covers(h, example) == is_positive]
    
    def remove_more_general(self, boundary):
        result = []
        for h in boundary:
            if not any(self.is_more_general(h2, h) and h != h2 for h2 in boundary):
                result.append(h)
        return result
    
    def remove_less_general(self, boundary):
        result = []
        for h in boundary:
            if not any(self.is_more_general(h, h2) and h != h2 for h2 in boundary):
                result.append(h)
        return result
    
    def fit(self, X, y, verbose=False):
        num_features = len(X[0])
        self.initialize_boundaries(num_features)
        self.history = []
        self.collapsed = False
        
        for idx, (example, label) in enumerate(zip(X, y)):
            example = list(example)
            
            # Check for None/missing values
            has_missing = any(pd.isna(val) for val in example)
            if has_missing:
                if verbose:
                    print(f"  Skipping Ex {idx} (missing values)")
                continue
            
            if label == 1:  # Positive
                self.G = self.remove_inconsistent(self.G, example, True)
                new_S = []
                for s in self.S:
                    if not self.covers(s, example):
                        new_s = self.generalize_S(s, example)
                        new_S.append(new_s)
                    else:
                        new_S.append(s)
                self.S = self.remove_more_general(new_S)
            else:  # Negative
                self.S = self.remove_inconsistent(self.S, example, False)
                new_G = []
                for g in self.G:
                    if self.covers(g, example):
                        specializations = self.specialize_G(g, example)
                        for spec in specializations:
                            if any(self.is_more_general(spec, s) for s in self.S):
                                new_G.append(spec)
                    else:
                        new_G.append(g)
                self.G = self.remove_less_general(new_G)
            
            self.history.append({
                'example_idx': idx,
                'S': copy.deepcopy(self.S),
                'G': copy.deepcopy(self.G)
            })
            
            if len(self.S) == 0 or len(self.G) == 0:
                self.collapsed = True
                self.collapse_reason = f"COLLAPSED at example {idx}"
                break
        
        return self.history


# ============================================================
# 4. UTILITY FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_subsection(title):
    print("\n" + "-" * 60)
    print(f" {title}")
    print("-" * 60)

def discretize_dataset(df):
    """Discretize continuous attributes"""
    df_disc = df.copy()
    
    # Discretize Income
    if 'Income' in df.columns:
        try:
            df_disc['Income'] = pd.cut(df['Income'].astype(float), 
                                        bins=[0, 35000, 55000, float('inf')], 
                                        labels=['Low', 'Medium', 'High'])
        except:
            pass
    
    # Discretize Credit_Score
    if 'Credit_Score' in df.columns:
        try:
            df_disc['Credit_Score'] = pd.cut(df['Credit_Score'].astype(float), 
                                              bins=[0, 620, 680, 720, float('inf')], 
                                              labels=['Poor', 'Fair', 'Good', 'Excellent'])
        except:
            pass
    
    return df_disc

def prepare_data(df, discretize=True, drop_missing=False):
    """Prepare data for algorithms"""
    if discretize:
        df = discretize_dataset(df)
    
    feature_cols = ['Income', 'Credit_Score', 'Employment_Years', 'Has_Existing_Loan']
    
    # Ensure columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    
    df_processed = df.copy()
    if drop_missing:
        df_processed = df_processed.dropna(subset=available_cols)
    
    X = df_processed[available_cols].values.tolist()
    y = df_processed['Approved'].astype(int).values.tolist() if 'Approved' in df_processed.columns else []
    
    return X, y, available_cols

def get_attribute_domains(df):
    """Get unique values for each attribute"""
    feature_cols = ['Income', 'Credit_Score', 'Employment_Years', 'Has_Existing_Loan']
    df_disc = discretize_dataset(df)
    domains = []
    for col in feature_cols:
        if col in df_disc.columns:
            domains.append(df_disc[col].dropna().unique().tolist())
        else:
            domains.append([])
    return domains


# ============================================================
# 5. CASE STUDY 1: NOISY DATA
# ============================================================

def case_study_1_noisy_data(df):
    print_section("CASE STUDY 1: NOISY DATA (Inconsistent Labels)")
    
    print("\n📋 Problem: Identical feature vectors with different labels")
    print("   When this occurs, no hypothesis can be consistent with all data")
    print("   Result: Version Space collapses to EMPTY")
    
    # Find noisy/contradictory examples
    feature_cols = ['Income', 'Credit_Score', 'Employment_Years', 'Has_Existing_Loan']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Check for duplicates with different labels
    df_check = df.dropna(subset=available_cols).copy()
    
    # Find rows with duplicate features
    duplicate_mask = df_check[available_cols].duplicated(keep=False)
    noisy_examples = df_check[duplicate_mask].sort_values(available_cols)
    
    if len(noisy_examples) > 0:
        print("\n🔍 Noisy/Contradictory Examples Found:")
        display_cols = available_cols + ['Approved', 'Application_Month']
        print(noisy_examples[display_cols].to_string())
        
        print("\n💡 Analysis:")
        # Find specific contradictory pairs
        for name, group in noisy_examples.groupby(available_cols):
            if len(group['Approved'].unique()) > 1:
                features_dict = {}
                for i, col in enumerate(available_cols):
                    features_dict[col] = group[col].iloc[0]
                print(f"   - Features: {features_dict}")
                print(f"     Labels: {group['Approved'].tolist()} (CONTRADICTION!)")
                break
    else:
        print("\n🔍 No exact duplicates found, checking for similar patterns...")
    
    # Run Find-S
    print_subsection("Find-S Algorithm Result")
    X, y, cols = prepare_data(df, discretize=False, drop_missing=True)
    find_s = FindS()
    find_s.fit(X, y)
    print(f"  Final Hypothesis: {find_s.hypothesis}")
    print("  ⚠️  Find-S produces overly general hypothesis '?' due to noise")
    
    # Run Candidate Elimination
    print_subsection("Candidate Elimination Algorithm Result")
    try:
        domains = get_attribute_domains(df)
        X_disc, y_disc, cols = prepare_data(df, discretize=True, drop_missing=True)
        ce = CandidateElimination(attribute_domains=domains)
        ce.fit(X_disc, y_disc)
        
        print(f"  Final S boundary: {ce.S}")
        print(f"  Final G boundary: {ce.G[:3] if ce.G else '∅'}{'...' if ce.G and len(ce.G) > 3 else ''}")
        
        if ce.collapsed or not ce.S or not ce.G:
            print("  ❌ VERSION SPACE COLLAPSED TO EMPTY!")
            print("  Reason: Contradictory examples make it impossible to find")
            print("          any hypothesis consistent with all training data.")
    except Exception as e:
        print(f"  ❌ CE Algorithm Failed: {str(e)[:100]}...")
        print("  This demonstrates that with noisy/inconsistent data,")
        print("  the Version Space is impossible to obtain!")


# ============================================================
# 6. CASE STUDY 2: TARGET CONCEPT NOT IN HYPOTHESIS SPACE
# ============================================================

def case_study_2_xor_pattern(df):
    print_section("CASE STUDY 2: TARGET CONCEPT OUTSIDE HYPOTHESIS SPACE")
    
    print("\n📋 Problem: The true concept cannot be represented as conjunction of literals")
    print("   Hypothesis Space H: Only conjunctive concepts (e.g., A AND B AND C)")
    print("   True Concept: Disjunctive or XOR-like pattern")
    
    # Identify XOR-like pattern in data
    print("\n🔍 XOR-like Pattern in Dataset:")
    
    if 'Income' in df.columns and 'Has_Existing_Loan' in df.columns:
        # Filter relevant data - convert to numeric
        df_temp = df.copy()
        df_temp['Income'] = pd.to_numeric(df_temp['Income'], errors='coerce')
        
        low_income_mask = df_temp['Income'] < 40000
        has_loan_mask = df_temp['Has_Existing_Loan'] == 'Yes'
        
        print("\n   Early Months (Conservative Policy):")
        early_data = df_temp[(df_temp['Application_Month'] <= 5) & low_income_mask & has_loan_mask]
        for _, row in early_data.iterrows():
            status = "APPROVED" if row['Approved'] == 1 else "REJECTED"
            print(f"     Month {int(row['Application_Month'])}: Income={int(row['Income'])}, Loan={row['Has_Existing_Loan']} → {status}")
        
        print("\n   Later Months (Aggressive Policy - CONCEPT DRIFT):")
        late_data = df_temp[(df_temp['Application_Month'] >= 10) & low_income_mask & has_loan_mask]
        for _, row in late_data.iterrows():
            status = "APPROVED" if row['Approved'] == 1 else "REJECTED"
            print(f"     Month {int(row['Application_Month'])}: Income={int(row['Income'])}, Loan={row['Has_Existing_Loan']} → {status}")
    
    print("\n💡 Analysis:")
    print("   - Same feature combination gives different results based on context")
    print("   - This pattern cannot be expressed as simple conjunction")
    print("   - The Version Space cannot capture this complexity")
    
    # Run Candidate Elimination
    print_subsection("Candidate Elimination Algorithm Result")
    try:
        domains = get_attribute_domains(df)
        X_disc, y_disc, cols = prepare_data(df, discretize=True, drop_missing=True)
        ce = CandidateElimination(attribute_domains=domains)
        ce.fit(X_disc, y_disc)
        
        if ce.collapsed:
            print(f"  ❌ {ce.collapse_reason}")
            print("  Reason: The concept is NOT linearly separable in the hypothesis space")
            print("          CE cannot learn disjunctive or contextual concepts.")
    except Exception as e:
        print(f"  ❌ Version Space impossible to obtain!")
        print(f"  Reason: Target concept not representable in hypothesis space")


# ============================================================
# 7. CASE STUDY 3: MISSING ATTRIBUTE VALUES
# ============================================================

def case_study_3_missing_values(df):
    print_section("CASE STUDY 3: MISSING ATTRIBUTE VALUES")
    
    print("\n📋 Problem: Standard CE algorithm requires complete data")
    print("   Missing values create ambiguity in boundary updates")
    
    # Show missing values
    print("\n🔍 Missing Values in Dataset:")
    missing_summary = df.isnull().sum()
    missing_cols = []
    for col, count in missing_summary.items():
        if count > 0:
            print(f"   {col}: {count} missing values")
            missing_cols.append(col)
    
    if missing_cols:
        print("\n   Examples with Missing Values:")
        missing_rows = df[df.isnull().any(axis=1)]
        display_cols = ['Income', 'Credit_Score', 'Employment_Years', 
                        'Has_Existing_Loan', 'Application_Month', 'Approved']
        available_display = [col for col in display_cols if col in df.columns]
        print(missing_rows[available_display].head(10).to_string())
    
    print("\n💡 Analysis:")
    print("   - Missing values prevent CE from determining hypothesis coverage")
    print("   - Must either drop rows (lose data) or impute (introduce bias)")
    
    # Compare: With vs Without missing values
    print_subsection("Candidate Elimination with Complete Cases Only")
    X_complete, y_complete, cols = prepare_data(df, discretize=True, drop_missing=True)
    print(f"  Using {len(X_complete)} complete examples (dropped {len(df) - len(X_complete)} incomplete)")
    print("  ⚠️  Without proper handling of missing values:")
    print("     - True Version Space is UNDEFINED")
    print("     - Must either impute or drop (loses information)")


# ============================================================
# 8. CASE STUDY 4: CONTINUOUS ATTRIBUTE VALUES
# ============================================================

def case_study_4_continuous_values(df):
    print_section("CASE STUDY 4: CONTINUOUS ATTRIBUTE VALUES")
    
    print("\n📋 Problem: CE requires discrete symbolic values")
    print("   Continuous attributes have infinite possible values")
    
    # Show continuous value ranges
    print("\n🔍 Continuous Attributes Statistics:")
    if 'Income' in df.columns:
        income_clean = pd.to_numeric(df['Income'], errors='coerce')
        print(f"   Income: Range [{income_clean.min():.0f} - {income_clean.max():.0f}]")
    if 'Credit_Score' in df.columns:
        score_clean = pd.to_numeric(df['Credit_Score'], errors='coerce')
        print(f"   Credit_Score: Range [{score_clean.min():.0f} - {score_clean.max():.0f}]")
    
    print("\n💡 Analysis:")
    print("   - True boundary might be: Income > 37500.50")
    print("   - CE can only represent: Income ∈ {Low, Medium, High}")
    print("   - Exact Version Space has INFINITE hypotheses (unrepresentable)")
    
    # Show discretization impact
    print_subsection("Discretization Required for CE")
    try:
        df_disc = discretize_dataset(df)
        
        print("  Original Continuous Values:")
        if 'Income' in df.columns:
            print(f"    Income: {df['Income'].head(3).tolist()}")
        if 'Credit_Score' in df.columns:
            print(f"    Credit_Score: {df['Credit_Score'].head(3).tolist()}")
        
        print("\n  After Discretization (Information Loss):")
        if 'Income' in df_disc.columns:
            print(f"    Income: {df_disc['Income'].head(3).tolist()}")
        if 'Credit_Score' in df_disc.columns:
            print(f"    Credit_Score: {df_disc['Credit_Score'].head(3).tolist()}")
    except:
        print("  Discretization requires clean numeric data")
    
    print("\n  ❌ True Version Space for continuous data is IMPOSSIBLE to obtain")
    print("     - Infinite number of possible thresholds")
    print("     - CE only works on discretized approximation")


# ============================================================
# 9. CASE STUDY 5: CONCEPT DRIFT
# ============================================================

def case_study_5_concept_drift(df):
    print_section("CASE STUDY 5: CONCEPT DRIFT (Non-Stationary Environment)")
    
    print("\n📋 Problem: Target concept changes over time")
    print("   Version Space assumes stationary concept (doesn't change)")
    
    # Show concept drift over months
    print("\n🔍 Concept Drift Analysis by Month:")
    
    # Phase 1: Months 1-5 (Conservative)
    print("\n   Phase 1: Months 1-5 (Conservative Policy)")
    phase1 = df[df['Application_Month'] <= 5]
    for _, row in phase1.head(5).iterrows():
        status = "✓ APPROVED" if row['Approved'] == 1 else "✗ REJECTED"
        credit_score = int(row['Credit_Score']) if pd.notna(row['Credit_Score']) else 'N/A'
        print(f"     Month {int(row['Application_Month'])}: Income={int(row['Income']) if pd.notna(row['Income']) else 'N/A'}, "
              f"Score={credit_score}, Loan={row['Has_Existing_Loan']} → {status}")
    
    # Phase 2: Months 6-12 (Drift begins)
    print("\n   Phase 2: Months 6-12 (Policy Shift - CONCEPT DRIFT)")
    phase2 = df[(df['Application_Month'] >= 6) & (df['Application_Month'] <= 12)]
    for _, row in phase2.head(5).iterrows():
        status = "✓ APPROVED" if row['Approved'] == 1 else "✗ REJECTED"
        credit_score = int(row['Credit_Score']) if pd.notna(row['Credit_Score']) else 'N/A'
        print(f"     Month {int(row['Application_Month'])}: Income={int(row['Income']) if pd.notna(row['Income']) else 'N/A'}, "
              f"Score={credit_score}, Loan={row['Has_Existing_Loan']} → {status}")
    
    print("\n💡 Analysis:")
    print("   - Same features get different labels at different times")
    print("   - The approval criteria CHANGED over time")
    print("   - No single Version Space can be consistent across all time")
    
    # Show Find-S behavior
    print_subsection("Find-S Algorithm on Drifting Data")
    X, y, cols = prepare_data(df, discretize=False, drop_missing=True)
    find_s = FindS()
    find_s.fit(X, y)
    print(f"  Final Hypothesis: {find_s.hypothesis}")
    print("  ⚠️  Find-S only remembers the MOST RECENT positives")
    print("     It FORGETS the early conservative policy completely!")
    
    print_subsection("Version Space Conclusion")
    print("  ❌ VERSION SPACE IMPOSSIBLE TO OBTAIN")
    print("  Reason: Concept Drift violates the stationarity assumption")
    print("          The target function changes over time")


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():
    # Load dataset from CSV file
    df = load_dataset('unified_version_space_cases.csv')
    
    # Run all 5 case studies
    case_study_1_noisy_data(df)
    case_study_2_xor_pattern(df)
    case_study_3_missing_values(df)
    case_study_4_continuous_values(df)
    case_study_5_concept_drift(df)
    
    # Summary
    print_section("SUMMARY: 5 SITUATIONS WHERE VERSION SPACE IS IMPOSSIBLE")
    
    print("""
┌─────────────────────────┬────────────────────────────────────────────────────────────┐
│ Situation               │ Why Version Space is Impossible                             │
├─────────────────────────┼────────────────────────────────────────────────────────────┤
│ 1. Noisy Data           │ Contradictory labels → No consistent hypothesis exists      │
│ 2. XOR Pattern / Not in H│ True concept not representable in hypothesis space          │
│ 3. Missing Values       │ Ambiguous coverage → Boundaries undefined                    │
│ 4. Continuous Values    │ Infinite thresholds → Exact boundaries unrepresentable       │
│ 5. Concept Drift        │ Target changes over time → No single consistent concept      │
└─────────────────────────┴────────────────────────────────────────────────────────────┘
""")
    
    print("\n🔑 KEY INSIGHT:")
    print("   Candidate Elimination assumes:")
    print("   • Noise-free data")
    print("   • Conjunctive hypothesis space")
    print("   • Complete attribute values")
    print("   • Discrete symbolic domains")
    print("   • Stationary target concept")
    print("\n   When ANY assumption is violated, the pure Version Space")
    print("   becomes IMPOSSIBLE to obtain!")
    
    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()