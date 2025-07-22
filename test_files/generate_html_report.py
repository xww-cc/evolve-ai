#!/usr/bin/env python3
"""
HTML报告生成器 - 将JSON数据转换为专业的HTML报告
"""

import json
import os
from datetime import datetime

def load_report_data(filename='model_evaluation_report.json'):
    """加载报告数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_html_report(data):
    """生成HTML报告"""
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evolve AI - 模型评估测试报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .test-info {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .test-info h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .info-item {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .info-item h3 {{
            font-size: 1.2em;
            margin-bottom: 10px;
        }}
        
        .info-item .value {{
            font-size: 1.8em;
            font-weight: bold;
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .metric-detail {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }}
        
        .performance-bar {{
            background: #ecf0f1;
            border-radius: 10px;
            height: 30px;
            margin: 10px 0;
            overflow: hidden;
        }}
        
        .performance-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .recommendations {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }}
        
        .recommendations h2 {{
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .recommendation-item {{
            background: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid white;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .info-grid, .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Evolve AI 模型评估测试报告</h1>
            <p>专业的人工智能进化系统性能评估报告</p>
        </div>
        
        <div class="test-info">
            <h2>📊 测试概览</h2>
            <div class="info-grid">
                <div class="info-item">
                    <h3>测试日期</h3>
                    <div class="value">{data['test_info']['test_date'][:10]}</div>
                </div>
                <div class="info-item">
                    <h3>总测试时间</h3>
                    <div class="value">{data['test_info']['total_test_time']:.2f}s</div>
                </div>
                <div class="info-item">
                    <h3>测试状态</h3>
                    <div class="value">✅ 完成</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚡ 性能指标</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>种群创建性能</h3>
                    <div class="metric-value">{data['performance_metrics']['population_creation']['individuals_per_second']:.0f}</div>
                    <div class="metric-detail">个体/秒</div>
                    <div class="metric-detail">平均时间: {data['performance_metrics']['population_creation']['average_time']:.3f}s</div>
                </div>
                
                <div class="metric-card">
                    <h3>系统稳定性</h3>
                    <div class="metric-value">{data['performance_metrics']['stability']['success_rate']*100:.1f}%</div>
                    <div class="metric-detail">成功率</div>
                    <div class="metric-detail">平均执行时间: {data['performance_metrics']['stability']['average_execution_time']:.3f}s</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔬 评估结果</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>符号推理评估</h3>
                    <div class="metric-value">{data['evaluation_results']['symbolic']['average_score']:.3f}</div>
                    <div class="metric-detail">平均得分 ± {data['evaluation_results']['symbolic']['std_score']:.3f}</div>
                    <div class="metric-detail">速度: {data['evaluation_results']['symbolic']['individuals_per_second']:.0f} 个体/秒</div>
                    <div class="metric-detail">范围: {data['evaluation_results']['symbolic']['min_score']:.3f} - {data['evaluation_results']['symbolic']['max_score']:.3f}</div>
                </div>
                
                <div class="metric-card">
                    <h3>真实世界评估</h3>
                    <div class="metric-value">{data['evaluation_results']['realworld']['average_score']:.3f}</div>
                    <div class="metric-detail">平均得分 ± {data['evaluation_results']['realworld']['std_score']:.3f}</div>
                    <div class="metric-detail">速度: {data['evaluation_results']['realworld']['individuals_per_second']:.0f} 个体/秒</div>
                    <div class="metric-detail">范围: {data['evaluation_results']['realworld']['min_score']:.3f} - {data['evaluation_results']['realworld']['max_score']:.3f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🔄 进化分析</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>进化性能</h3>
                    <div class="metric-value">{data['evolution_analysis']['generations_per_second']:.0f}</div>
                    <div class="metric-detail">代/秒</div>
                    <div class="metric-detail">平均进化时间: {data['evolution_analysis']['average_evolution_time']:.3f}s</div>
                </div>
                
                <div class="metric-card">
                    <h3>种群多样性</h3>
                    <div class="metric-value">{data['evolution_analysis']['average_diversity']:.3f}</div>
                    <div class="metric-detail">平均多样性 ± {data['evolution_analysis']['std_diversity']:.3f}</div>
                    <div class="metric-detail">多样性维护率: {data['diversity_metrics']['diversity_maintenance_rate']*100:.1f}%</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 可扩展性分析</h2>
            <div class="chart-container">
                <h3>不同种群大小的性能表现</h3>
                <div class="metrics-grid">
    """
    
    # 添加可扩展性数据
    scalability = data['performance_metrics']['scalability']
    for size, metrics in scalability.items():
        html += f"""
                    <div class="metric-card">
                        <h3>种群大小 {size}</h3>
                        <div class="metric-value">{metrics['time_per_individual']*1000:.1f}ms</div>
                        <div class="metric-detail">每个体处理时间</div>
                        <div class="metric-detail">创建: {metrics['creation_time']:.3f}s</div>
                        <div class="metric-detail">评估: {metrics['evaluation_time']:.3f}s</div>
                    </div>
        """
    
    html += """
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🌊 多样性分析</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>初始多样性</h3>
                    <div class="metric-value">{data['diversity_metrics']['average_initial_diversity']:.3f}</div>
                    <div class="metric-detail">平均初始多样性</div>
                </div>
                
                <div class="metric-card">
                    <h3>最终多样性</h3>
                    <div class="metric-value">{data['diversity_metrics']['average_final_diversity']:.3f}</div>
                    <div class="metric-detail">平均最终多样性</div>
                </div>
                
                <div class="metric-card">
                    <h3>多样性变化</h3>
                    <div class="metric-value">{data['diversity_metrics']['average_diversity_change']:.3f}</div>
                    <div class="metric-detail">平均变化量</div>
                </div>
            </div>
        </div>
    """
    
    # 添加建议部分
    if data['recommendations']:
        html += """
        <div class="recommendations">
            <h2>💡 改进建议</h2>
        """
        for i, rec in enumerate(data['recommendations'], 1):
            html += f"""
            <div class="recommendation-item">
                <strong>{i}.</strong> {rec}
            </div>
            """
        html += """
        </div>
        """
    else:
        html += """
        <div class="recommendations">
            <h2>🎉 系统状态优秀</h2>
            <div class="recommendation-item">
                所有性能指标均达到预期标准，系统运行状态优秀，无需额外改进建议。
            </div>
        </div>
        """
    
    html += """
        <div class="footer">
            <p>📊 报告生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>🧠 Evolve AI - 让AI通过进化变得更智能</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html

def main():
    """主函数"""
    # 加载数据
    data = load_report_data()
    
    # 生成HTML
    html_content = generate_html_report(data)
    
    # 保存HTML文件
    with open('model_evaluation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML报告已生成: model_evaluation_report.html")
    print("📊 报告包含以下内容:")
    print("   - 测试概览")
    print("   - 性能指标")
    print("   - 评估结果")
    print("   - 进化分析")
    print("   - 可扩展性分析")
    print("   - 多样性分析")
    print("   - 改进建议")

if __name__ == "__main__":
    main() 