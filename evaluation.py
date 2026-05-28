from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
import numpy as np


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):


    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            if num_ratings == 1:
                num_ratings += 0.0000001
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    if denominator <= 0.0000001:
        denominator = 0.0000001
    return np.round((1.0 - numerator / denominator),4)






def Quadratic_Weighted_Kappa(y_true, y_pred, total_score):
    num_classes = int(total_score)+1
    
    conf_matrix = np.zeros((num_classes, num_classes))
    for n in range(num_classes):
        i_c = int(y_true[n])
        j_c = int(y_pred[n])
        conf_matrix[i_c, j_c] += 1
    
    
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2)/((num_classes-1) ** 2)
    
    obs_sum = np.sum(conf_matrix)
    exp_sum = np.outer(np.sum(conf_matrix, axis=1), np.sum(conf_matrix, axis=0)) / obs_sum
    
    nom = np.sum(weights * conf_matrix)
    denom = np.sum(weights * exp_sum)
    if denom <= 0.0000001:
        denom = 0.0000001
    quadratic_kappa = 1 - (nom / denom)
    
    return quadratic_kappa


# ============ 统计显著性检验函数 ============
from scipy.stats import ttest_rel, wilcoxon
import pandas as pd

def statistical_significance_test(scores_adaptive, scores_fixed, method='ttest'):
    """
    对自适应权重vs固定权重的5折结果进行配对检验
    
    Args:
        scores_adaptive: list of QWK scores from 5 folds (adaptive DEG)
        scores_fixed: list of QWK scores from 5 folds (fixed weights)
        method: 't-test' or 'wilcoxon'
    
    Returns:
        dict with t_stat, p_value, significance_level
    """
    assert len(scores_adaptive) == len(scores_fixed) == 5, \
        "Need exactly 5 fold results"
    
    if method == 'ttest':
        t_stat, p_val = ttest_rel(scores_adaptive, scores_fixed)
        
    elif method == 'wilcoxon':
        # 非参数检验（对异常值更鲁棒）
        t_stat, p_val = wilcoxon(scores_adaptive, scores_fixed)
    
    # 确定显著性水平
    if p_val < 0.001:
        sig_level = "***"
        interpretation = "极显著"
    elif p_val < 0.01:
        sig_level = "**"
        interpretation = "显著"
    elif p_val < 0.05:
        sig_level = "*"
        interpretation = "弱显著"
    else:
        sig_level = "ns"
        interpretation = "不显著"
    
    result = {
        'method': method,
        't_statistic': round(t_stat, 4),
        'p_value': round(p_val, 4),
        'significance': sig_level,
        'interpretation': interpretation,
        'mean_adaptive': round(np.mean(scores_adaptive), 4),
        'mean_fixed': round(np.mean(scores_fixed), 4),
        'improvement': round(np.mean(scores_adaptive) - np.mean(scores_fixed), 4),
        'std_adaptive': round(np.std(scores_adaptive), 4),
        'std_fixed': round(np.std(scores_fixed), 4),
    }
    
    return result


def generate_significance_report(results_dict, output_file=None):
    """
    生成统计显著性报告表
    
    Args:
        results_dict: {'model_name': {'adaptive': [scores], 'fixed': [scores]}, ...}
        output_file: 保存报告的文件路径
    """
    report_data = []
    
    for model_name, scores in results_dict.items():
        test_result = statistical_significance_test(
            scores['adaptive'], 
            scores['fixed']
        )
        
        row = {
            'Model': model_name,
            'Adaptive_QWK': f"{test_result['mean_adaptive']}±{test_result['std_adaptive']}",
            'Fixed_QWK': f"{test_result['mean_fixed']}±{test_result['std_fixed']}",
            'Improvement': f"{test_result['improvement']}",
            't-statistic': test_result['t_statistic'],
            'p-value': test_result['p_value'],
            'Significance': test_result['significance'],
        }
        report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    
    if output_file:
        report_df.to_csv(output_file, index=False)
        print(f"Report saved to {output_file}")
    
    return report_df


def print_significance_summary(results_dict):
    """
    打印统计显著性总结到控制台
    """
    print("\n" + "="*80)
    print("统计显著性检验报告 (5-Fold Cross-Validation)")
    print("="*80)
    
    for model_name, scores in results_dict.items():
        test_result = statistical_significance_test(scores['adaptive'], scores['fixed'])
        
        print(f"\n【{model_name}】")
        print(f"  自适应DEG:  {test_result['mean_adaptive']} ± {test_result['std_adaptive']}")
        print(f"  固定权重:   {test_result['mean_fixed']} ± {test_result['std_fixed']}")
        print(f"  改进幅度:   {test_result['improvement']} ({test_result['improvement']/test_result['mean_fixed']*100:.2f}%)")
        print(f"  t统计量:    {test_result['t_statistic']}")
        print(f"  p值:       {test_result['p_value']}")
        print(f"  显著性:     {test_result['significance']} ({test_result['interpretation']})")
    
    print("\n" + "="*80)
    print("注: *** p<0.001, ** p<0.01, * p<0.05, ns=不显著")
    print("="*80)