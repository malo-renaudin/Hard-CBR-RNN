import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def prepare_data_correct(rt_data, surprisal_data):
    """
    Prepare data exactly as specified in paper
    """
    # Merge and filter
    data = rt_data.merge(surprisal_data, on=['item', 'zone'], how='inner', suffixes=('_rt', '_surprisal'))
    data = data[(data['correct'] >= 5) & (data['RT'] >= 100) & (data['RT'] <= 3000)]
    data = data.sort_values(['item', 'zone']).reset_index(drop=True)
    
    # Handle word column
    if 'word_surprisal' in data.columns:
        data['word'] = data['word_surprisal']
    elif 'word_rt' in data.columns:
        data['word'] = data['word_rt']
    elif 'word' in data.columns:
        pass
    else:
        print("Available columns:", data.columns.tolist())
        raise ValueError("Cannot find word column after merge")
    
    # Log transform reading times
    data['log_RT'] = np.log(data['RT'])
    
    # Create predictors
    data['word_length'] = data['word'].str.len()
    data['word_position'] = data['zone'] 
    
    # Unigram surprisal
    word_counts = data['word'].value_counts()
    total_words = len(data)
    data['unigram_surprisal'] = data['word'].map(lambda w: -np.log(word_counts.get(w, 1) / total_words))
    
    # Current and previous word surprisal
    data['current_surprisal'] = data['surprisal']
    data['prev_surprisal'] = data.groupby('item')['surprisal'].shift(1)
    
    # Remove first words
    data = data.dropna(subset=['prev_surprisal']).reset_index(drop=True)
    
    # Subject coding
    data['subject'] = data['WorkerId'].astype('category')
    data['item'] = data['item'].astype('category')
    
    print(f"Final data: {len(data)} observations, {data['subject'].nunique()} subjects")
    return data


def compute_delta_loglik_correct(data):
    """
    Proper mixed-effects modeling with correct statsmodels syntax
    """
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    
    print("\n" + "="*60)
    print("FITTING MODELS FOR ΔLOGLIK CALCULATION")
    print("="*60)
    
    print(f"Data shape: {len(data)} observations")
    print("Dependent variable: log(RT)")
    
    try:
        # Statsmodels mixed-effects syntax is different from R
        print("\nFitting baseline model...")
        baseline_model = smf.mixedlm(
            "log_RT ~ word_length + word_position + unigram_surprisal", 
            data, 
            groups=data['subject']
        ).fit(method=['nm'])
        
        print("Fitting full model...")
        full_model = smf.mixedlm(
            "log_RT ~ word_length + word_position + unigram_surprisal + current_surprisal + prev_surprisal", 
            data, 
            groups=data['subject']
        ).fit(method=['nm'])
        
        baseline_ll = baseline_model.llf
        full_ll = full_model.llf
        delta_loglik = full_ll - baseline_ll
        
    except Exception as e:
        print(f"Mixed-effects modeling failed: {e}")
        print("Falling back to OLS with clustered standard errors...")
        
        # Convert categorical to dummy codes for OLS
        data_ols = data.copy()
        data_ols['subject_numeric'] = pd.Categorical(data_ols['subject']).codes
        
        # Fallback with clustered standard errors
        baseline_model = smf.ols(
            "log_RT ~ word_length + word_position + unigram_surprisal", 
            data_ols
        ).fit(cov_type='cluster', cov_kwds={'groups': data_ols['subject_numeric']})
        
        full_model = smf.ols(
            "log_RT ~ word_length + word_position + unigram_surprisal + current_surprisal + prev_surprisal", 
            data_ols
        ).fit(cov_type='cluster', cov_kwds={'groups': data_ols['subject_numeric']})
        
        baseline_ll = baseline_model.llf
        full_ll = full_model.llf
        delta_loglik = full_ll - baseline_ll
    
    # Results
    print(f"\n" + "="*60)
    print("ΔLOGLIK RESULTS")
    print("="*60)
    print(f"Baseline Log-Likelihood: {baseline_ll:.2f}")
    print(f"Full Model Log-Likelihood: {full_ll:.2f}")  
    print(f"ΔLogLik: {delta_loglik:.2f}")
    
    if np.isfinite(delta_loglik):
        lr_stat = 2 * delta_loglik
        df_diff = 2  # Adding current_surprisal + prev_surprisal
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
        
        print(f"Likelihood Ratio Test: χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.6f}")
        
        if delta_loglik > 0:
            print(f"✓ Surprisal improves model fit by {delta_loglik:.2f} log-likelihood units")
        else:
            print(f"✗ Surprisal does not improve model fit (ΔLogLik = {delta_loglik:.2f})")
    
    return {
        'delta_loglik': delta_loglik,
        'baseline_model': baseline_model,
        'full_model': full_model,
        'lr_test': {
            'statistic': lr_stat if 'lr_stat' in locals() else np.nan,
            'p_value': p_value if 'p_value' in locals() else np.nan,
            'df': 2
        }
    }

def run_paper_analysis_correct(rt_data, surprisal_data):
    """
    Run the exact analysis from the paper to get ΔLogLik
    """
    
    print("="*60)  
    print("PAPER-COMPLIANT ΔLOGLIK ANALYSIS")
    print("="*60)
    
    # Prepare data
    print("Preparing data...")
    data = prepare_data_correct(rt_data, surprisal_data)
    
    # Compute ΔLogLik
    results = compute_delta_loglik_correct(data)
    
    # Summary
    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"ΔLogLik for your CBR-RNN: {results['delta_loglik']:.2f}")
    
    if np.isfinite(results['delta_loglik']):
        if results['delta_loglik'] > 0:
            print("🎉 Your model improves prediction of human reading times!")
            print(f"Improvement: {results['delta_loglik']:.2f} log-likelihood units")
        else:
            print("❌ Your model does not improve reading time prediction")
    else:
        print("⚠ Numerical issues prevented ΔLogLik calculation")
    
    return results

