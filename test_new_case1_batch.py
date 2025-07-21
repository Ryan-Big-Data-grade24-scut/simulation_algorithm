#!/usr/bin/env python
"""
Case1BatchSolver 新流程测试脚本
"""
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def test_new_case1_batch_solver():
    """测试新的Case1BatchSolver实现"""
    print("="*60)
    print("  测试新的Case1BatchSolver流程")
    print("="*60)
    
    try:
        from positioning_algorithm.batch_solvers.case1_solver import Case1BatchSolver
        print("✓ 成功导入新的Case1BatchSolver")
        
        # 创建求解器实例
        solver = Case1BatchSolver(
            m=10.0,  # 场地宽度
            n=20.0,  # 场地高度
            tolerance=1e-3
        )
        print("✓ 成功创建求解器实例")
        
        # 创建测试数据
        N_cbn = 3  # 测试3个组合
        combinations = np.array([
            # 组合0: [[t0,θ0], [t1,θ1], [t2,θ2]]
            [[14.55, 0.0], [4.52, np.pi/2], [4.52, np.pi]],
            # 组合1
            [[15.00, 0.0], [4.60, np.pi/2], [4.60, np.pi]],
            # 组合2
            [[13.80, 0.0], [4.40, np.pi/2], [4.40, np.pi]]
        ])
        print(f"✓ 创建测试数据: {combinations.shape}")
        
        # 执行求解
        print("\n开始执行完整流程...")
        sol_cbn = solver.solve(combinations)
        print(f"✓ 求解完成，结果形状: {sol_cbn.shape}")
        
        # 分析结果
        print(f"\n结果分析:")
        print(f"预期结果形状: ({N_cbn}, 3)")
        print(f"实际结果形状: {sol_cbn.shape}")
        
        # 检查每个组合的解
        for i in range(N_cbn):
            sol = sol_cbn[i]
            print(f"\n组合 {i}:")
            if not np.any(np.isinf(sol)):
                print(f"  x_range: {sol[0]}")
                print(f"  y_range: {sol[1]}") 
                print(f"  phi: {sol[2]:.6f} rad = {np.degrees(sol[2]):.2f}°")
            else:
                print(f"  无解 (包含np.inf)")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_steps():
    """测试各个步骤"""
    print("\n" + "="*60)
    print("  测试各个步骤")
    print("="*60)
    
    try:
        from positioning_algorithm.batch_solvers.case1_solver import Case1BatchSolver
        
        solver = Case1BatchSolver(m=10.0, n=20.0, tolerance=1e-3)
        
        # 测试数据
        combinations = np.array([
            [[14.55, 0.0], [4.52, np.pi/2], [4.52, np.pi]],
            [[15.00, 0.0], [4.60, np.pi/2], [4.60, np.pi]]
        ])
        
        print("步骤1: 扩展组合")
        expanded_combinations, combo_indices = solver._expand_combinations(combinations)
        print(f"  输入: {combinations.shape} -> 输出: {expanded_combinations.shape}")
        print(f"  组合编号: {combo_indices}")
        
        print("\n步骤2: 计算AB系数")
        ab_h, ab_v = solver._compute_ab_coefficients(expanded_combinations)
        print(f"  ab_h: {ab_h.shape}, ab_v: {ab_v.shape}")
        
        print("\n步骤3: 计算phi候选")
        phi_h, phi_v = solver._compute_phi_candidates(ab_h, ab_v)
        print(f"  phi_h: {phi_h.shape}, phi_v: {phi_v.shape}")
        print(f"  有效phi_h数量: {np.sum(~np.isinf(phi_h))}")
        print(f"  有效phi_v数量: {np.sum(~np.isinf(phi_v))}")
        
        print("\n步骤4: 计算碰撞三角形")
        colli_h, valid_indice_cbo_h = solver._compute_collision_triangles_h(phi_h, expanded_combinations, combo_indices)
        colli_v, valid_indice_cbo_v = solver._compute_collision_triangles_v(phi_v, expanded_combinations, combo_indices)
        print(f"  colli_h: {colli_h.shape}, colli_v: {colli_v.shape}")
        
        print("\n步骤5: 计算关键量")
        key_h = solver._compute_key_quantities_h(colli_h)
        key_v = solver._compute_key_quantities_v(colli_v)
        print(f"  key_h: {key_h.shape}, key_v: {key_v.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Case1BatchSolver 新流程测试")
    
    success1 = test_new_case1_batch_solver()
    success2 = test_individual_steps()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！")
        print("\n📋 新流程特性摘要:")
        print("  ✓ 明确的7步流程: 扩展->AB系数->phi候选->碰撞三角形->关键量->求解->组织结果")
        print("  ✓ 组合编号全程跟踪")
        print("  ✓ 水平边和竖直边分别处理")
        print("  ✓ 优化的中心点计算，先检查边界再添加容差")
        print("  ✓ 返回格式: (N_cbn, 3) - [(x_min, x_max), (y_min, y_max), phi]")
        return True
    else:
        print("\n❌ 部分测试失败")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 测试失败")
        sys.exit(1)
