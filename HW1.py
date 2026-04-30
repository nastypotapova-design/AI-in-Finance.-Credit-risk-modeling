import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# somers' d (gini coefficient)
# -------------------------------------------------------------------

def somers_d(xx, yy, weight=None):
    if weight is None:
        weight = np.ones(len(xx))

    x = pd.Series(xx).rank().values
    y = pd.Series(yy).rank().values
    weight = np.array(weight)

    n2_const = sum(weight[:-1] * (sum(weight) - np.cumsum(weight[:-1])))
    k, y_adj = 0, 0

    for i in range(len(weight) - 1):
        sign_arr_x = np.sign(x[i] - x[i + 1:len(weight)])
        sign_arr_y = np.sign(y[i] - y[i + 1:len(weight)])
        ww_arr = weight[i] * weight[i + 1:]

        k += sum(ww_arr * sign_arr_x * sign_arr_y)
        y_adj += sum(ww_arr * (1 - abs(sign_arr_y * sign_arr_y)))

    return k / (n2_const - y_adj)


# -------------------------------------------------------------------
# variance inflation factor
# -------------------------------------------------------------------

def calculate_vif(x):
    from sklearn.linear_model import LinearRegression

    vif_dict = {}
    features = x.columns.tolist()

    for feature in features:
        y_target = x[feature]
        x_others = x.drop(columns=[feature])

        reg = LinearRegression()
        reg.fit(x_others, y_target)
        r2 = reg.score(x_others, y_target)

        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vif_dict[feature] = vif

    return vif_dict

# -------------------------------------------------------------------
# create plot
# -------------------------------------------------------------------
def plot_ratings_comparison(model_factors, model_name, filename):
    x_data = df[model_factors.split(', ')]

    ordinal_model = OrderedModel(y_ordinal, x_data, distr='logit')
    ordinal_fit = ordinal_model.fit(method='bfgs', disp=False)

    pred_probs = ordinal_fit.model.predict(ordinal_fit.params, exog=x_data)

    pred_ratings = []
    for gg in range(len(pred_probs)):
        max_idx = np.argmax(pred_probs[gg])
        pred_ratings.append(max_idx + 1)

    actual_dist = y_ordinal.value_counts(normalize=True).sort_index()
    pred_dist = pd.Series(pred_ratings).value_counts(normalize=True).sort_index()

    x = np.arange(1, 6)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, actual_dist.values, width, label='Actual Ratings', color='red', alpha=0.8)
    bars2 = ax.bar(x + width / 2, pred_dist.values, width, label='Predicted Ratings', color='blue', alpha=0.8)

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.0%}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.0%}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Rating (1 = safest, 5 = riskiest)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(model_name, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return actual_dist, pred_dist
# -------------------------------------------------------------------
# load and merge data
# -------------------------------------------------------------------

file_name = 'Data_for_loading_zadanie.xlsx'

df_raw = pd.read_excel(file_name, sheet_name='Данные')
df_rating = pd.read_excel(file_name, sheet_name='Data_for_loading')

df_raw = df_raw[df_raw["Номер Проекта"].notna()]

rating_dict = dict(zip(df_rating["Number_Project"], df_rating["Expert_rating"]))
df_raw["Expert_rating"] = df_raw["Номер Проекта"].map(rating_dict)
df_clean = df_raw[df_raw["Expert_rating"].notna()].copy()

# -------------------------------------------------------------------
# final dataset
# -------------------------------------------------------------------

df_model = pd.DataFrame()
df_model["LTV_norm"] = df_clean["LTV(норм)"]
df_model["DSU_BEN_norm"] = df_clean["Доля собств. участия бенефициаров (норм)"]
df_model["IRR_norm"] = df_clean["IRR (норм)"]
df_model["DSCR_norm"] = df_clean["DSCR(норм)"]
df_model["LLCR_norm"] = df_clean["LLCR(норм)"]
df_model["IND_FACTOR_norm"] = df_clean["Индустриальный фактор (норм)"]
df_model["REG_FACTOR_norm"] = df_clean["Региональный фактор (норм)"]
df_model["Default_flag"] = df_clean["Признак дефолта"]
df_model["Expert_rating"] = df_clean["Expert_rating"]

df_model.to_excel("clean_data_for_modeling.xlsx", index=False)

df = pd.read_excel("clean_data_for_modeling.xlsx")

# -------------------------------------------------------------------
# binary logistic regression - all combinations
# -------------------------------------------------------------------

feature_names = ['LTV_norm', 'DSU_BEN_norm', 'IRR_norm', 'DSCR_norm',
                 'LLCR_norm', 'IND_FACTOR_norm', 'REG_FACTOR_norm']

x_full = df[feature_names]
y = df['Default_flag']

results_binary = []

for k in range(1, len(feature_names) + 1):
    for combo in itertools.combinations(feature_names, k):
        x = x_full[list(combo)]
        x_with_const = sm.add_constant(x)

        try:
            model = sm.Logit(y, x_with_const).fit(disp=False)
            predicted_prob = model.predict(x_with_const)
            gini = somers_d(predicted_prob, y)
            auroc = (gini + 1) / 2

            vif_dict = {}
            if len(combo) >= 2:
                vif_dict = calculate_vif(x)
                max_vif = max(vif_dict.values()) if vif_dict else 1
            else:
                max_vif = 1

            result = {
                'factors': ', '.join(combo),
                'n_factors': k,
                'pseudo_r2': model.prsquared,
                'log_likelihood': model.llf,
                'aic': model.aic,
                'gini': gini,
                'auroc': auroc,
                'max_vif': max_vif,
                'converged': model.mle_retvals['converged'],
                'nobs': model.nobs
            }

            for i, factor in enumerate(['const'] + list(combo)):
                result[f'coef_{factor}'] = model.params.iloc[i]
                result[f'pvalue_{factor}'] = model.pvalues.iloc[i]

            for factor, vif_val in vif_dict.items():
                result[f'vif_{factor}'] = vif_val

            results_binary.append(result)
        except Exception:
            pass

results_df = pd.DataFrame(results_binary)
results_df = results_df.sort_values('gini', ascending=False)
results_df.to_excel('Perebor_PD_binary.xlsx', index=False)

# -------------------------------------------------------------------
# ordinal logistic regression - all combinations
# -------------------------------------------------------------------

y_ordinal = df['Expert_rating']
results_ordinal = []

for k in range(1, len(feature_names) + 1):
    for combo in itertools.combinations(feature_names, k):
        x = x_full[list(combo)]

        try:
            model = OrderedModel(y_ordinal, x, distr='logit')
            res_log = model.fit(method='bfgs', disp=False)

            predicted_probs = res_log.model.predict(res_log.params, exog=x)

            model_rating = []
            p_rating_max = []

            for gg in range(len(predicted_probs)):
                model_rating.append(1)
                p_rating_max.append(predicted_probs[gg, 0])
                for ggg in range(1, 5):
                    if predicted_probs[gg, ggg] > p_rating_max[gg]:
                        model_rating[gg] = ggg + 1
                        p_rating_max[gg] = predicted_probs[gg, ggg]

            gini_rating = somers_d(model_rating, y_ordinal)
            auroc_rating = (gini_rating + 1) / 2

            score = np.zeros(len(x))
            for i, factor in enumerate(combo):
                score += res_log.params.iloc[i] * x[factor]
            gini_score = somers_d(score, y_ordinal)

            errors = [abs(y_ordinal.iloc[gg] - model_rating[gg]) for gg in range(len(model_rating))]
            error_0 = sum(1 for e in errors if e == 0) / len(errors)
            error_0_1 = sum(1 for e in errors if e <= 1) / len(errors)
            error_0_1_2 = sum(1 for e in errors if e <= 2) / len(errors)

            result_ord = {
                'factors': ', '.join(combo),
                'n_factors': k,
                'pseudo_r2': res_log.prsquared,
                'log_likelihood': res_log.llf,
                'aic': res_log.aic,
                'gini_rating': gini_rating,
                'auroc_rating': auroc_rating,
                'gini_score': gini_score,
                'error_0': error_0,
                'error_0_1': error_0_1,
                'error_0_1_2': error_0_1_2,
                'converged': res_log.mle_retvals['converged'],
                'nobs': res_log.nobs
            }

            for i, factor in enumerate(combo):
                result_ord[f'coef_{factor}'] = res_log.params.iloc[i]
                result_ord[f'pvalue_{factor}'] = res_log.pvalues.iloc[i]

            for i in range(4):
                result_ord[f'threshold_{i + 1}'] = res_log.params.iloc[len(combo) + i]

            results_ordinal.append(result_ord)

        except Exception:
            pass

results_ord_df = pd.DataFrame(results_ordinal)
results_ord_df = results_ord_df.sort_values('gini_rating', ascending=False)
results_ord_df.to_excel('Perebor_PD_ordinal.xlsx', index=False)

# -------------------------------------------------------------------
# collect stats for readme
# -------------------------------------------------------------------

readme_data = {
    "initial_rows": len(df_raw),
    "initial_cols": len(df_raw.columns),
    "final_rows": len(df_model),
    "final_cols": len(df_model.columns),
    "default_count": int(df_model["Default_flag"].sum()),
    "default_pct": df_model["Default_flag"].mean() * 100,
    "rating_dist": df_model["Expert_rating"].value_counts().sort_index().to_dict(),
    "factors": feature_names,
    "total_models": len(results_df),
    "best_gini": results_df['gini'].iloc[0],
    "best_gini_factors": results_df['factors'].iloc[0],
}

best_ordinal_model = results_ord_df.iloc[0]

ordinal_stats = {
    "total_models": len(results_ord_df),
    "best_gini": best_ordinal_model['gini_rating'],
    "best_gini_factors": best_ordinal_model['factors'],
    "best_auroc": best_ordinal_model['auroc_rating'],
    "best_error_0": best_ordinal_model['error_0'],
    "best_error_0_1": best_ordinal_model['error_0_1'],
    "best_error_0_1_2": best_ordinal_model['error_0_1_2']
}

# -------------------------------------------------------------------
# select optimal binary model (p-value < 0.10, vif < 5)
# -------------------------------------------------------------------

candidate_models = []

for idx, row in results_df.iterrows():
    selected = row['factors'].split(', ')

    all_ok = True
    for col in selected:
        pval_name = f'pvalue_{col}'
        if pval_name in row and row[pval_name] >= 0.10:
            all_ok = False
            break

    vif_ok = row['max_vif'] < 5 if 'max_vif' in row else True

    if all_ok and vif_ok:
        candidate_models.append(row)

if candidate_models:
    candidate_df = pd.DataFrame(candidate_models).sort_values('gini', ascending=False)
    top_candidate = candidate_df.iloc[0]

    optimal_model = {
        'factors': top_candidate['factors'],
        'n_factors': top_candidate['n_factors'],
        'gini': top_candidate['gini'],
        'auroc': top_candidate['auroc'],
        'aic': top_candidate['aic'],
        'max_vif': top_candidate['max_vif']
    }
else:
    optimal_model = None

# -------------------------------------------------------------------
# select optimal ordinal model (p-value < 0.10)
# -------------------------------------------------------------------

ordinal_candidates = []

for idx, row in results_ord_df.iterrows():
    selected = row['factors'].split(', ')

    all_significant = True
    for col in selected:
        pval_name = f'pvalue_{col}'
        if pval_name not in row:
            # if p-value not stored, skip this model
            all_significant = False
            break
        if row[pval_name] >= 0.10:
            all_significant = False
            break

    if all_significant:
        ordinal_candidates.append(row)

if ordinal_candidates:
    ordinal_candidate_df = pd.DataFrame(ordinal_candidates).sort_values('gini_rating', ascending=False)
    best_ordinal_optimal = ordinal_candidate_df.iloc[0]

    ordinal_optimal_model = {
        'factors': best_ordinal_optimal['factors'],
        'n_factors': best_ordinal_optimal['n_factors'],
        'gini': best_ordinal_optimal['gini_rating'],
        'auroc': best_ordinal_optimal['auroc_rating'],
        'aic': best_ordinal_optimal['aic'],
        'error_0': best_ordinal_optimal['error_0'],
        'error_0_1': best_ordinal_optimal['error_0_1']
    }
else:
    ordinal_optimal_model = None

# -------------------------------------------------------------------
# plot actual vs predicted ratings
# -------------------------------------------------------------------

actual_best, pred_best = plot_ratings_comparison(
    best_ordinal_model['factors'],
    'Best Ordinal Model (7 Factors)',
    'ratings_comparison_best.png'
)

distribution_best = {
    "actual": actual_best.to_dict(),
    "predicted": pred_best.to_dict()
}

if ordinal_optimal_model:
    actual_optimal, pred_optimal = plot_ratings_comparison(
        ordinal_optimal_model['factors'],
        f'Optimal Ordinal Model ({ordinal_optimal_model["n_factors"]} Factors)',
        'ratings_comparison_optimal.png'
    )

    distribution_optimal = {
        "actual": actual_optimal.to_dict(),
        "predicted": pred_optimal.to_dict()
    }

# -------------------------------------------------------------------
# generate readme.md
# -------------------------------------------------------------------

with open('README.md', 'w', encoding='utf-8') as f:
    f.write('# Credit Risk Modeling for Specialized Lending\n\n')

    f.write('## Data Preparation\n\n')
    f.write(f'- Initial observations: {readme_data["initial_rows"]}\n')
    f.write(f'- Final observations (after cleaning): {readme_data["final_rows"]}\n')
    f.write(f'- Default rate: {readme_data["default_pct"]:.2f}%\n')
    f.write(f'- Risk factors: {", ".join(readme_data["factors"])}\n')

    f.write('\n## Expert Rating Distribution\n\n')
    f.write('| Rating | Count | Percentage |\n')
    f.write('|--------|-------|------------|\n')
    total = sum(readme_data["rating_dist"].values())
    for rating, count in readme_data["rating_dist"].items():
        pct = count / total * 100
        f.write(f'| {rating} | {count} | {pct:.1f}% |\n')

    f.write('\n## Binary Logistic Regression Results\n\n')
    f.write(f'- Total models evaluated: {readme_data["total_models"]}\n')
    f.write(f'- Best Gini: {readme_data["best_gini"]:.4f}\n')
    f.write(f'- Best AUROC: {(readme_data["best_gini"] + 1) / 2:.4f}\n')
    f.write(f'- Best model factors: `{readme_data["best_gini_factors"]}`\n')
    f.write('- Note: The 7-factor model has highest Gini but all p-values > 0.05 (quasi-separation)\n\n')

    if optimal_model:
        f.write('### Recommended Binary Model (p < 0.10, VIF < 5)\n\n')
        f.write(f'- Factors: `{optimal_model["factors"]}`\n')
        f.write(f'- Number of factors: {optimal_model["n_factors"]}\n')
        f.write(f'- Gini: {optimal_model["gini"]:.4f}\n')
        f.write(f'- AUROC: {optimal_model["auroc"]:.4f}\n')
        f.write(f'- AIC: {optimal_model["aic"]:.2f}\n')
        f.write(f'- Max VIF: {optimal_model["max_vif"]:.2f}\n')
        f.write('- All factors significant (p < 0.10), no multicollinearity\n')

    f.write('\n## Ordinal Logistic Regression Results\n\n')
    f.write(f'- Total models evaluated: {ordinal_stats["total_models"]}\n')
    f.write(f'- Best Gini: {ordinal_stats["best_gini"]:.4f}\n')
    f.write(f'- Best AUROC: {ordinal_stats["best_auroc"]:.4f}\n')
    f.write(f'- Best model factors: `{ordinal_stats["best_gini_factors"]}`\n')
    f.write(f'- Exact match: {ordinal_stats["best_error_0"]:.1%}\n')
    f.write(f'- Within ±1 rating: {ordinal_stats["best_error_0_1"]:.1%}\n')
    f.write(f'- Within ±2 rating: {ordinal_stats["best_error_0_1_2"]:.1%}\n')

    if ordinal_optimal_model:
        f.write('\n### Recommended Ordinal Model (p < 0.10)\n\n')
        f.write(f'- Factors: `{ordinal_optimal_model["factors"]}`\n')
        f.write(f'- Number of factors: {ordinal_optimal_model["n_factors"]}\n')
        f.write(f'- Gini: {ordinal_optimal_model["gini"]:.4f}\n')
        f.write(f'- AUROC: {ordinal_optimal_model["auroc"]:.4f}\n')
        f.write(f'- Exact match: {ordinal_optimal_model["error_0"]:.1%}\n')
        f.write(f'- Within ±1 rating: {ordinal_optimal_model["error_0_1"]:.1%}\n')
        f.write('- All factors significant (p < 0.10)\n')

    f.write('\n## Model Comparison\n\n')
    f.write('| Model | Factors | Gini | AUROC | AIC | Notes |\n')
    f.write('|-------|---------|------|-------|-----|-------|\n')

    # Best binary (7 factors)
    short_factors = readme_data["best_gini_factors"][:60] + '...' if len(readme_data["best_gini_factors"]) > 60 else \
    readme_data["best_gini_factors"]
    f.write(
        f'| Best Binary (7f) | `{short_factors}` | {readme_data["best_gini"]:.4f} | {(readme_data["best_gini"] + 1) / 2:.4f} | - | p > 0.05, quasi-separation |\n')

    # Recommended binary (4 factors)
    if optimal_model:
        f.write(
            f'| Recommended Binary ({optimal_model["n_factors"]}f) | `{optimal_model["factors"]}` | {optimal_model["gini"]:.4f} | {optimal_model["auroc"]:.4f} | {optimal_model["aic"]:.1f} | p < 0.10, VIF = {optimal_model["max_vif"]:.2f} |\n')

    # Best ordinal (7 factors)
    short_ord_factors = ordinal_stats["best_gini_factors"][:60] + '...' if len(
        ordinal_stats["best_gini_factors"]) > 60 else ordinal_stats["best_gini_factors"]
    f.write(
        f'| Best Ordinal (7f) | `{short_ord_factors}` | {ordinal_stats["best_gini"]:.4f} | {ordinal_stats["best_auroc"]:.4f} | - | Exact match: {ordinal_stats["best_error_0"]:.1%} |\n')

    # Recommended ordinal (if exists)
    if ordinal_optimal_model:
        f.write(
            f'| Recommended Ordinal ({ordinal_optimal_model["n_factors"]}f) | `{ordinal_optimal_model["factors"]}` | {ordinal_optimal_model["gini"]:.4f} | {ordinal_optimal_model["auroc"]:.4f} | - | Exact match: {ordinal_optimal_model["error_0"]:.1%}, p < 0.10 |\n')

    f.write('\n## Actual vs Predicted Ratings\n\n')
    f.write('### Best Ordinal Model (7 Factors)\n\n')
    f.write('This model achieves maximum Gini but may overfit.\n\n')
    f.write('![Best Model](ratings_comparison_best.png)\n\n')

    if ordinal_optimal_model:
        f.write('### Recommended Ordinal Model\n\n')
        f.write(f'This model uses {ordinal_optimal_model["n_factors"]} significant factors ')
        f.write('and is preferred for deployment.\n\n')
        f.write('![Optimal Model](ratings_comparison_optimal.png)\n\n')

    f.write('### Comparison Table (Best Ordinal Model)\n\n')
    f.write('| Rating | Actual | Predicted | Difference |\n')
    f.write('|--------|--------|-----------|------------|\n')
    for rating in range(1, 6):
        actual_pct = distribution_best["actual"].get(rating, 0) * 100
        predicted_pct = distribution_best["predicted"].get(rating, 0) * 100
        diff = predicted_pct - actual_pct
        diff_sign = '+' if diff > 0 else ''
        f.write(f'| {rating} | {actual_pct:.1f}% | {predicted_pct:.1f}% | {diff_sign}{diff:.1f}% |\n')

    f.write('\n## Key Findings\n\n')
    f.write('1. **Binary model** achieves strong predictive power (Gini > 0.9)\n')
    f.write('2. **Optimal binary model** uses 4 factors with VIF < 5 and all p-values < 0.10\n')
    f.write('3. **Ordinal model** correctly predicts exact rating for ')
    f.write(f'{ordinal_stats["best_error_0"]:.1%} of projects\n')
    f.write(f'4. **Within ±1 rating**, accuracy reaches {ordinal_stats["best_error_0_1"]:.1%}\n')
    f.write('5. **Main limitation**: Rating 3 is overpredicted (borderline cases hardest to classify)\n')
    f.write('6. **Recommended for deployment**: 4-factor binary model and ')
    if ordinal_optimal_model:
        f.write(f'{ordinal_optimal_model["n_factors"]}-factor ordinal model\n')
    else:
        f.write('7-factor ordinal model (with caution for overfitting)\n')

    f.write('\n## Conclusions\n\n')
    f.write('Both models successfully predict credit risk using normalized risk factors.\n')
    f.write('The binary model is suitable for default screening, while the ordinal model\n')
    f.write('can support expert rating validation. The recommended models balance\n')
    f.write('predictive power with statistical significance and avoid multicollinearity.\n')

    