import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def prepare_data_correct(rt_data, surprisal_data):
    """
    Prepare data exactly as specified in paper
    """
    # Merge and filter - handle duplicate column names
    data = rt_data.merge(surprisal_data, on=['item', 'zone'], how='inner', suffixes=('_rt', '_surprisal'))
    data = data[(data['correct'] >= 5) & (data['RT'] >= 100) & (data['RT'] <= 3000)]
    data = data.sort_values(['item', 'zone']).reset_index(drop=True)
    
    # Use the word column from surprisal data (should be the same)
    if 'word_surprisal' in data.columns:
        data['word'] = data['word_surprisal']
    elif 'word_rt' in data.columns:
        data['word'] = data['word_rt']
    elif 'word' in data.columns:
        pass  # Already have word column
    else:
        # Check what columns we actually have
        print("Available columns:", data.columns.tolist())
        raise ValueError("Cannot find word column after merge")
    
    # Create exact predictors from paper
    data['word_length'] = data['word'].str.len()
    data['word_position'] = data['zone'] 
    
    # Unigram surprisal - fix the duplicate index issue
    word_counts = data['word'].value_counts()
    total_words = len(data)
    data['unigram_surprisal'] = data['word'].map(lambda w: -np.log(word_counts.get(w, 1) / total_words))
    
    # Current and previous word surprisal
    data['current_surprisal'] = data['surprisal']
    data['prev_surprisal'] = data.groupby('item')['surprisal'].shift(1)
    
    # Remove first words (no previous surprisal)
    data = data.dropna(subset=['prev_surprisal']).reset_index(drop=True)
    
    # Subject coding
    data['subject'] = pd.Categorical(data['WorkerId']).codes
    
    print(f"Final data: {len(data)} observations, {data['subject'].nunique()} subjects")
    return data


def fit_simple_lmer(y, X, groups, max_attempts=3):
    """
    Robust LME fitting with fallbacks to handle numerical issues
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    import statsmodels.api as sm
    
    # Clean data
    mask = ~(np.isnan(y) | np.isnan(X).any(axis=1) | np.isnan(groups))
    y_clean = y[mask]
    X_clean = X[mask]
    groups_clean = groups[mask]
    
    attempts = [
        ("MixedLM with random intercepts", lambda: MixedLM(y_clean, X_clean, groups=groups_clean).fit(method='nm', maxiter=100)),
        ("MixedLM with BFGS", lambda: MixedLM(y_clean, X_clean, groups=groups_clean).fit(method='bfgs', maxiter=50)),
        ("OLS fallback", lambda: sm.OLS(y_clean, X_clean).fit())
    ]
    
    for name, fit_func in attempts[:max_attempts]:
        try:
            result = fit_func()
            if hasattr(result, 'llf') and np.isfinite(result.llf):
                print(f"  Fitted with {name}, LogLik = {result.llf:.2f}")
                return result
            elif hasattr(result, 'llf'):
                print(f"  {name} converged but LogLik infinite")
        except Exception as e:
            print(f"  {name} failed: {str(e)[:50]}...")
            continue
    
    raise Exception("All fitting methods failed")

def compute_delta_loglik_correct(data):
    """
    Compute ΔLogLik exactly as in paper:
    Compare baseline vs baseline + current_surprisal + prev_surprisal
    """
    
    print("\n" + "="*60)
    print("FITTING MODELS FOR ΔLOGLIK CALCULATION")
    print("="*60)
    
    # Prepare design matrices
    y = data['RT'].values  # Raw RT as specified
    
    # Baseline: intercept + word_length + word_position + unigram_surprisal
    X_baseline = np.column_stack([
        np.ones(len(data)),
        data['word_length'].values,
        data['word_position'].values, 
        data['unigram_surprisal'].values
    ])
    
    # Full: baseline + current_surprisal + prev_surprisal
    X_full = np.column_stack([
        X_baseline,
        data['current_surprisal'].values,
        data['prev_surprisal'].values
    ])
    
    groups = data['subject'].values
    
    print(f"Data shape: {len(y)} observations")
    print(f"Baseline predictors: {X_baseline.shape[1]} (intercept + word_length + word_position + unigram_surprisal)")
    print(f"Full predictors: {X_full.shape[1]} (baseline + current_surprisal + prev_surprisal)")
    
    # Fit baseline model
    print("\nFitting baseline model...")
    baseline_result = fit_simple_lmer(y, X_baseline, groups)
    
    # Fit full model  
    print("\nFitting full model...")
    full_result = fit_simple_lmer(y, X_full, groups)
    
    # Calculate ΔLogLik
    baseline_ll = baseline_result.llf
    full_ll = full_result.llf
    delta_loglik = full_ll - baseline_ll
    
    print(f"\n" + "="*60)
    print("ΔLOGLIK RESULTS (PAPER SPECIFICATION)")
    print("="*60)
    print(f"Baseline Log-Likelihood: {baseline_ll:.2f}")
    print(f"Full Model Log-Likelihood: {full_ll:.2f}")  
    print(f"ΔLogLik: {delta_loglik:.2f}")
    
    if np.isfinite(delta_loglik):
        # Likelihood ratio test
        lr_stat = 2 * delta_loglik
        df_diff = X_full.shape[1] - X_baseline.shape[1]  # Should be 2 (current + prev surprisal)
        p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
        
        print(f"Likelihood Ratio Test: χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_value:.6f}")
        significance = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
        print(f"Significance: {significance}")
        
        # Effect interpretation
        if delta_loglik > 0:
            print(f"✓ Your CBR-RNN surprisal improves model fit by {delta_loglik:.2f} log-likelihood units")
        else:
            print(f"✗ Surprisal does not improve model fit (ΔLogLik = {delta_loglik:.2f})")
    else:
        print("⚠ ΔLogLik calculation failed due to numerical issues")
        print("Models fitted but log-likelihood values are problematic")
    
    # Extract surprisal coefficients from full model
    print(f"\n" + "="*40)
    print("SURPRISAL COEFFICIENTS")
    print("="*40)
    
    if hasattr(full_result, 'params') and len(full_result.params) >= 6:
        current_coef = full_result.params[4]  # 5th coefficient (current surprisal)
        prev_coef = full_result.params[5]     # 6th coefficient (previous surprisal)
        
        if hasattr(full_result, 'pvalues'):
            current_p = full_result.pvalues[4]
            prev_p = full_result.pvalues[5]
        else:
            current_p = prev_p = np.nan
            
        print(f"Current word surprisal: {current_coef:.4f} (p = {current_p:.4f})")
        print(f"Previous word surprisal: {prev_coef:.4f} (p = {prev_p:.4f})")
        
        if current_coef > 0:
            print("✓ Current surprisal has expected positive effect")
        else:
            print("⚠ Current surprisal has unexpected negative effect")
    
    return {
        'delta_loglik': delta_loglik,
        'baseline_model': baseline_result,
        'full_model': full_result,
        'lr_test': {
            'statistic': lr_stat if 'lr_stat' in locals() else np.nan,
            'p_value': p_value if 'p_value' in locals() else np.nan,
            'df': df_diff if 'df_diff' in locals() else 2
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

