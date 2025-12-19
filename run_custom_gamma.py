"""
사용자 정의 Gamma 값으로 실행하는 스크립트
Custom Gamma values execution script

사용법 / Usage:
    python3 run_custom_gamma.py toy 0 5 10          # Gamma = 0, 5, 10 (toy instance)
    python3 run_custom_gamma.py full 0 20 40 60     # Gamma = 0, 20, 40, 60 (full instance)
"""

import sys
import time
from datetime import datetime
from DH_config import ProblemConfig
from DH_data_gen import SupplyChainData
from DH_algo import CCGAlgorithm
from DH_main import save_results, plot_sensitivity_results

def run_custom_sensitivity(instance_type, gamma_values):
    """
    사용자 지정 Gamma 값들로 sensitivity analysis 실행

    Args:
        instance_type: 'toy' or 'full'
        gamma_values: list of int, Gamma 값들
    """
    print("="*80)
    print("사용자 정의 Gamma Sensitivity Analysis")
    print("="*80)
    print(f"Instance: {instance_type}")
    print(f"Gamma values: {gamma_values}")
    print("="*80)
    print()

    # Load data
    config = ProblemConfig(instance_type=instance_type)
    data = SupplyChainData.load(config.data_file)
    data.print_summary()

    # Results storage
    results = []

    # Run for each Gamma value
    for gamma in gamma_values:
        print()
        print("="*80)
        print(f"실행 중: Γ = {gamma}")
        print("="*80)

        config_gamma = ProblemConfig(instance_type=instance_type)
        config_gamma.Gamma = gamma

        ccg = CCGAlgorithm(data, config_gamma)
        result = ccg.run()

        # Store result
        result['Gamma'] = gamma
        results.append(result)

        print()
        print(f"✓ Γ = {gamma} 완료")
        print(f"  최적값: {result['optimal_value']:.2f}")
        print(f"  수렴: {result['converged']}")
        print(f"  반복: {result['iterations']}")
        print(f"  시간: {result['total_time']:.2f}s")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = save_results(results, instance_type, timestamp)

    # Plot
    print()
    print("그래프 생성 중...")
    png_filename = plot_sensitivity_results(results, instance_type, timestamp)

    print()
    print("="*80)
    print("완료!")
    print("="*80)
    print(f"결과 CSV: {csv_filename}")
    print(f"결과 그래프: {png_filename}")
    print("="*80)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법:")
        print("  python3 run_custom_gamma.py <instance_type> <gamma1> <gamma2> ...")
        print()
        print("예시:")
        print("  python3 run_custom_gamma.py toy 0 5 10")
        print("  python3 run_custom_gamma.py full 0 20 40 60 80 100")
        sys.exit(1)

    instance_type = sys.argv[1]
    if instance_type not in ['toy', 'full']:
        print(f"Error: instance_type은 'toy' 또는 'full'이어야 합니다. (입력값: {instance_type})")
        sys.exit(1)

    # Parse gamma values
    try:
        gamma_values = [int(g) for g in sys.argv[2:]]
    except ValueError:
        print("Error: Gamma 값은 정수여야 합니다.")
        sys.exit(1)

    # Run
    results = run_custom_sensitivity(instance_type, gamma_values)
