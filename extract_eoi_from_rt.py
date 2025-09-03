"""
Complete SAP Benchmark Log-RT Analysis for All Constructions
Following SAP Benchmark methodology (Huang et al. 2024, Appendix A)
Handles all construction types: Garden Path, Agreement, Relative Clauses, Attachment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mixedlm
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def prepare_full_dataset(df):
    """
    Prepare complete dataset following SAP Benchmark preprocessing
    """
    print("=== FULL DATASET PREPROCESSING ===")
    print("Following SAP Benchmark methodology...")
    
    data = df.copy()
    
    # Basic info
    print(f"Original data shape: {data.shape}")
    print(f"Constructions: {sorted(data['CONSTRUCTION'].unique())}")
    print(f"Types: {sorted(data['Type'].unique())}")
    
    # 1. Filter extreme RTs (SAP Benchmark: 100-7000ms)
    print(f"\nRT filtering...")
    print(f"RT range before: {data['RT'].min()}-{data['RT'].max()}ms")
    
    # Handle negative RTs (data quality issue)
    if data['RT'].min() < 0:
        print(f"⚠️  Found {len(data[data['RT'] < 0]):,} negative RTs - removing")
        data = data[data['RT'] >= 0]
    
    data = data[(data['RT'] >= 100) & (data['RT'] <= 7000)].copy()
    print(f"After RT filtering: {data.shape}")
    print(f"RT range after: {data['RT'].min()}-{data['RT'].max()}ms")
    
    # 2. Create log-transformed RT
    data['log_RT'] = np.log(data['RT'])
    
    # 3. Create participant and item IDs
    data['participant_id'] = pd.Categorical(data['MD5'])
    data['item_id'] = pd.Categorical(data['item'])
    
    # 4. Define regions of interest using ROI column
    data['region'] = 'other'
    data.loc[data['ROI'] == 0, 'region'] = 'critical'
    data.loc[data['ROI'] == 1, 'region'] = 'spillover1'
    data.loc[data['ROI'] == 2, 'region'] = 'spillover2'
    
    # 5. Filter to critical regions and experimental conditions
    experimental_constructions = ['NPZ', 'MVRR', 'NPS', 'RelativeClause', 'Attachment', 'NP/Z Agreement']
    critical_data = data[
        (data['CONSTRUCTION'].isin(experimental_constructions)) &
        (data['region'].isin(['critical', 'spillover1', 'spillover2']))
    ].copy()
    
    print(f"\nCritical regions data:")
    print(f"  Total observations: {len(critical_data):,}")
    print(f"  Participants: {critical_data['participant_id'].nunique()}")
    print(f"  Items: {critical_data['item_id'].nunique()}")
    print(f"  Constructions: {sorted(critical_data['CONSTRUCTION'].unique())}")
    
    return critical_data

def prepare_construction_subsets(data):
    """
    Prepare individual construction subsets with proper contrast coding
    """
    print("\n=== PREPARING CONSTRUCTION SUBSETS ===")
    
    subsets = {}
    
    # 1. Garden Path Constructions (3 types)
    garden_path_constructions = {
        'MVRR': ['MVRR_AMB', 'MVRR_UAMB'],  # Main Verb/Reduced Relative
        'NPZ': ['NPZ_AMB', 'NPZ_UAMB'],      # Transitive/Intransitive  
        'NPS': ['NPS_AMB', 'NPS_UAMB']       # Direct Object/Sentential Complement
    }
    
    for construction, types in garden_path_constructions.items():
        subset = data[data['Type'].isin(types)].copy()
        if len(subset) > 0:
            # Ambiguity coding: Unambiguous = 0, Ambiguous = 1
            subset['ambiguity'] = subset['Type'].str.contains('AMB').astype(int)
            subset['ambiguity'] = subset['ambiguity'] - subset['Type'].str.contains('UAMB').astype(int)
            subset['ambiguity'] = (subset['ambiguity'] > 0).astype(int)
            
            subsets[f'{construction}_garden_path'] = subset
            print(f"{construction} Garden Path: {len(subset):,} observations")
    
    # 2. Agreement Violation
    agreement_types = ['AGREE', 'AGREE_UNG']
    agreement_subset = data[data['Type'].isin(agreement_types)].copy()
    if len(agreement_subset) > 0:
        # Grammaticality coding: Grammatical = 0, Ungrammatical = 1
        agreement_subset['grammaticality'] = (agreement_subset['Type'] == 'AGREE_UNG').astype(int)
        subsets['agreement'] = agreement_subset
        print(f"Agreement: {len(agreement_subset):,} observations")
    
    # 3. Relative Clauses
    rc_types = ['RC_SUB', 'RC_OBJ']
    rc_subset = data[data['Type'].isin(rc_types)].copy()
    if len(rc_subset) > 0:
        # RC type coding: Subject RC = 0, Object RC = 1
        rc_subset['rc_type'] = (rc_subset['Type'] == 'RC_OBJ').astype(int)
        subsets['relative_clause'] = rc_subset
        print(f"Relative Clause: {len(rc_subset):,} observations")
    
    # 4. Attachment Ambiguity
    attach_types = ['AttachHigh', 'AttachLow', 'AttachMulti']
    attach_subset = data[data['Type'].isin(attach_types)].copy()
    if len(attach_subset) > 0:
        # Create contrast coding for attachment
        attach_subset['attachment'] = attach_subset['Type'].map({
            'AttachHigh': 0,    # High attachment
            'AttachLow': 1,     # Low attachment  
            'AttachMulti': 2    # Ambiguous
        })
        subsets['attachment'] = attach_subset
        print(f"Attachment: {len(attach_subset):,} observations")
    
    # Print summary
    print(f"\nTotal subsets prepared: {len(subsets)}")
    for name, subset in subsets.items():
        print(f"  {name}: {len(subset):,} observations, {subset['participant_id'].nunique()} participants")
    
    return subsets

def fit_construction_model(data, effect_var, construction_name, region='spillover1'):
    """
    Fit mixed-effects model for any construction type
    """
    print(f"\n=== FITTING {construction_name.upper()} MODEL: {region.upper()} ===")
    
    # Filter to specific region
    region_data = data[data['region'] == region].copy()
    
    if len(region_data) == 0:
        print(f"❌ No data for region {region}")
        return None, None
    
    print(f"Region data: {len(region_data):,} observations")
    print(f"Participants: {region_data['participant_id'].nunique()}")
    print(f"Items: {region_data['item_id'].nunique()}")
    
    # Calculate baseline RT
    baseline_rt = np.exp(region_data['log_RT'].mean())
    print(f"Baseline RT: {baseline_rt:.1f}ms")
    
    # Model fitting with fallback strategy
    model_info = {
        'construction': construction_name,
        'region': region,
        'n_obs': len(region_data),
        'n_participants': region_data['participant_id'].nunique(),
        'n_items': region_data['item_id'].nunique(),
        'baseline_rt': baseline_rt
    }
    
    try:
        # Attempt maximal model
        print("🔄 Attempting maximal random effects...")
        model = mixedlm(
            f"log_RT ~ {effect_var}",
            groups=region_data["participant_id"],
            re_formula=effect_var,
            vc_formula={"item_id": f"0 + {effect_var}"},
            data=region_data
        )
        result = model.fit(method='powell', maxiter=2000)
        model_info['type'] = 'maximal'
        print("✅ Maximal model converged!")
        
    except:
        try:
            # Fallback: Random intercepts only
            print("🔄 Attempting random intercepts...")
            model = mixedlm(
                f"log_RT ~ {effect_var}",
                groups=region_data["participant_id"],
                vc_formula={"item_id": "1"},
                data=region_data
            )
            result = model.fit(method='powell', maxiter=2000)
            model_info['type'] = 'random_intercepts'
            print("✅ Random intercepts model converged!")
            
        except:
            try:
                # Final fallback: Participant only
                print("🔄 Attempting participant-only...")
                model = mixedlm(
                    f"log_RT ~ {effect_var}",
                    groups=region_data["participant_id"],
                    data=region_data
                )
                result = model.fit(method='powell', maxiter=2000)
                model_info['type'] = 'participant_only'
                print("✅ Participant-only model converged!")
                
            except Exception as e:
                print(f"❌ All models failed: {str(e)[:100]}")
                return None, None
    
    return result, model_info

def extract_effect_from_model(model_result, effect_var, model_info):
    """
    Extract effect of interest from log-RT model and convert to milliseconds
    """
    # Get coefficient
    log_effect = model_result.params[effect_var]
    log_se = model_result.bse[effect_var]
    
    # Convert to multiplicative effect
    multiplicative_effect = np.exp(log_effect)
    
    # Convert to milliseconds
    baseline_rt = model_info['baseline_rt']
    ms_effect = baseline_rt * (multiplicative_effect - 1)
    
    # Confidence intervals
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, model_result.df_resid)
    log_ci_lower = log_effect - t_critical * log_se
    log_ci_upper = log_effect + t_critical * log_se
    
    ms_ci_lower = baseline_rt * (np.exp(log_ci_lower) - 1)
    ms_ci_upper = baseline_rt * (np.exp(log_ci_upper) - 1)
    
    # Statistical significance
    t_stat = log_effect / log_se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), model_result.df_resid))
    
    significance_stars = ('***' if p_value < 0.001 else 
                         '**' if p_value < 0.01 else 
                         '*' if p_value < 0.05 else 'ns')
    
    return {
        'construction': model_info['construction'],
        'region': model_info['region'],
        'log_effect': log_effect,
        'log_se': log_se,
        'multiplicative_effect': multiplicative_effect,
        'ms_effect': ms_effect,
        'ms_ci_lower': ms_ci_lower,
        'ms_ci_upper': ms_ci_upper,
        'baseline_rt': baseline_rt,
        't_stat': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'significance_stars': significance_stars,
        'model_info': model_info
    }

def analyze_all_constructions(subsets):
    """
    Analyze all constructions across all regions
    """
    print("\n" + "="*80)
    print("COMPLETE SAP BENCHMARK ANALYSIS - ALL CONSTRUCTIONS")
    print("="*80)
    
    regions = ['critical', 'spillover1', 'spillover2']
    all_results = {}
    
    # Define effect variables for each construction type
    effect_configs = {
        'MVRR_garden_path': {'var': 'ambiguity', 'name': 'MVRR Garden Path'},
        'NPZ_garden_path': {'var': 'ambiguity', 'name': 'NPZ Garden Path'},  
        'NPS_garden_path': {'var': 'ambiguity', 'name': 'NPS Garden Path'},
        'agreement': {'var': 'grammaticality', 'name': 'Agreement Violation'},
        'relative_clause': {'var': 'rc_type', 'name': 'Object vs Subject RC'},
        'attachment': {'var': 'attachment', 'name': 'Attachment Ambiguity'}
    }
    
    for subset_name, subset_data in subsets.items():
        if subset_name not in effect_configs:
            print(f"⚠️  Skipping {subset_name} - no configuration found")
            continue
            
        config = effect_configs[subset_name]
        construction_results = {}
        
        print(f"\n{'='*20} {config['name'].upper()} {'='*20}")
        
        for region in regions:
            print(f"\n--- {region.upper()} REGION ---")
            
            # Fit model
            model_result, model_info = fit_construction_model(
                subset_data, config['var'], config['name'], region
            )
            
            if model_result is not None:
                # Extract effect
                effect = extract_effect_from_model(model_result, config['var'], model_info)
                
                construction_results[region] = {
                    'model': model_result,
                    'effect': effect
                }
                
                # Print summary
                print(f"Effect: {effect['ms_effect']:>6.1f}ms "
                      f"[{effect['ms_ci_lower']:>5.1f}, {effect['ms_ci_upper']:>5.1f}] "
                      f"{effect['significance_stars']}")
            else:
                construction_results[region] = None
        
        all_results[subset_name] = construction_results
    
    return all_results

def create_comprehensive_plot(results):
    """
    Create comprehensive plot showing all construction effects
    """
    print("\n📈 Creating comprehensive visualization...")
    
    # Prepare plot data
    plot_data = []
    
    for construction, regions in results.items():
        for region, result in regions.items():
            if result is not None:
                effect = result['effect']
                plot_data.append({
                    'construction': effect['construction'],
                    'region': region,
                    'effect': effect['ms_effect'],
                    'ci_lower': effect['ms_ci_lower'],
                    'ci_upper': effect['ms_ci_upper'],
                    'significant': effect['significant'],
                    'p_value': effect['p_value']
                })
    
    if not plot_data:
        print("❌ No data to plot")
        return None
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create subplot for each construction
    constructions = plot_df['construction'].unique()
    n_constructions = len(constructions)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    regions = ['critical', 'spillover1', 'spillover2']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, construction in enumerate(constructions):
        if i >= len(axes):
            break
            
        ax = axes[i]
        construction_data = plot_df[plot_df['construction'] == construction]
        
        # Plot data for this construction
        effects = []
        ci_lowers = []
        ci_uppers = []
        significances = []
        
        for region in regions:
            region_data = construction_data[construction_data['region'] == region]
            if len(region_data) > 0:
                effects.append(region_data['effect'].iloc[0])
                ci_lowers.append(region_data['ci_lower'].iloc[0])
                ci_uppers.append(region_data['ci_upper'].iloc[0])
                significances.append(region_data['significant'].iloc[0])
            else:
                effects.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
                significances.append(False)
        
        # Create bars
        bars = ax.bar(range(len(regions)), effects, color=colors, alpha=0.7)
        
        # Error bars
        ax.errorbar(range(len(regions)), effects,
                   yerr=[np.array(effects) - np.array(ci_lowers),
                         np.array(ci_uppers) - np.array(effects)],
                   fmt='none', color='black', capsize=5)
        
        # Significance stars
        for j, (effect, sig) in enumerate(zip(effects, significances)):
            if sig and effect != 0:
                height = effect + (ci_uppers[j] - effects[j]) + 2
                ax.text(j, height, '*', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title(construction, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=45)
        ax.set_ylabel('Effect Size (ms)')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(constructions), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('SAP Benchmark Effects - All Constructions\n(Log-RT Models)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return fig

def create_summary_table(results):
    """
    Create comprehensive summary table
    """
    print("\n" + "="*100)
    print("📋 COMPREHENSIVE SUMMARY - ALL CONSTRUCTIONS")
    print("="*100)
    
    # Collect all results
    table_data = []
    
    for construction, regions in results.items():
        for region, result in regions.items():
            if result is not None:
                effect = result['effect']
                model_info = effect['model_info']
                
                table_data.append({
                    'Construction': effect['construction'],
                    'Region': region.title(),
                    'N_obs': f"{model_info['n_obs']:,}",
                    'Effect_ms': f"{effect['ms_effect']:.1f}",
                    'CI_lower': f"{effect['ms_ci_lower']:.1f}",
                    'CI_upper': f"{effect['ms_ci_upper']:.1f}",
                    'p_value': f"{effect['p_value']:.4f}",
                    'Sig': effect['significance_stars']
                })
    
    if table_data:
        summary_df = pd.DataFrame(table_data)
        
        # Print formatted table
        print(f"{'Construction':<20} {'Region':<12} {'N_obs':<10} {'Effect':<8} {'95% CI':<18} {'p-value':<10} {'Sig':<4}")
        print("-" * 100)
        
        for _, row in summary_df.iterrows():
            ci_str = f"[{row['CI_lower']}, {row['CI_upper']}]"
            print(f"{row['Construction']:<20} {row['Region']:<12} {row['N_obs']:<10} "
                  f"{row['Effect_ms']:<8} {ci_str:<18} {row['p_value']:<10} {row['Sig']:<4}")
    
    return table_data

def run_complete_sap_analysis(df):
    """
    Run complete SAP Benchmark analysis for all constructions
    """
    print("🚀 COMPLETE SAP BENCHMARK LOG-RT ANALYSIS")
    print("Following SAP Benchmark Appendix A methodology")
    print("="*80)
    
    # 1. Prepare full dataset
    data = prepare_full_dataset(df)
    
    # 2. Create construction subsets
    subsets = prepare_construction_subsets(data)
    
    if not subsets:
        print("❌ No valid construction subsets found")
        return None, None
    
    # 3. Analyze all constructions
    results = analyze_all_constructions(subsets)
    
    # 4. Create visualizations
    plot = create_comprehensive_plot(results)
    
    # 5. Create summary table
    summary = create_summary_table(results)
    
    # 6. Final summary
    print(f"\n✅ COMPLETE ANALYSIS FINISHED!")
    print(f"📊 Constructions analyzed: {len(results)}")
    
    significant_effects = []
    for construction, regions in results.items():
        for region, result in regions.items():
            if result is not None and result['effect']['significant']:
                effect_size = result['effect']['ms_effect']
                construction_name = result['effect']['construction']
                significant_effects.append(f"{construction_name} ({region}): {effect_size:+.1f}ms")
    
    if significant_effects:
        print(f"📈 Significant effects found:")
        for effect in significant_effects:
            print(f"   • {effect}")
    else:
        print(f"📊 No significant effects found across constructions")
    
    print(f"\n📝 Note: All effects from log(RT) models, back-transformed to milliseconds")
    print(f"📚 Methodology: SAP Benchmark (Huang et al. 2024, Appendix A)")
    
    return results, data

# Example usage:
# results, processed_data = run_complete_sap_analysis(your_full_dataframe)