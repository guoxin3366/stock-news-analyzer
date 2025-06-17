import os
os.environ['HF_MIRROR'] = 'https://mirror.sjtu.edu.cn/hugging-face-models'

import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import time
import json
from datetime import datetime

print("股票新闻AI分析工具启动...")

# 使用国内镜像加速下载
print("正在初始化AI模型...")
sentiment_analyzer = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

def fetch_finance_news():
    """获取财经新闻"""
    print("正在获取最新财经新闻...")
    sources = {
        "新浪财经": "https://finance.sina.com.cn/roll/index.d.html?cid=56589",
        "东方财富": "https://finance.eastmoney.com/"
    }
    
    all_news = []
    
    for source_name, url in sources.items():
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if "sina" in url:
                items = soup.select('.listBlk li')[:5]
                for item in items:
                    title = item.a.text.strip()
                    link = item.a['href']
                    if not link.startswith('http'):
                        link = 'https:' + link
                    all_news.append({"source": source_name, "title": title, "link": link})
            
            elif "eastmoney" in url:
                items = soup.select('.news-list li')[:5]
                for item in items:
                    title = item.a.text.strip()
                    link = item.a['href']
                    if not link.startswith('http'):
                        link = 'https://finance.eastmoney.com' + link
                    all_news.append({"source": source_name, "title": title, "link": link})
        
        except Exception as e:
            print(f"从 {source_name} 获取新闻失败: {str(e)}")
    
    return all_news

def analyze_news(news):
    """分析单条新闻"""
    try:
        print(f"分析新闻: {news['title']}")
        response = requests.get(news["link"], timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取正文内容
        content = ""
        if "sina" in news["link"]:
            content_div = soup.select_one('.article') or soup.select_one('#artibody')
            if content_div:
                content = content_div.get_text().strip()[:1000]
        elif "eastmoney" in news["link"]:
            content_div = soup.select_one('.newsContent') or soup.select_one('#ContentBody')
            if content_div:
                content = content_div.get_text().strip()[:1000]
        
        if not content:
            content = news['title']
        
        # 情感分析
        result = sentiment_analyzer(content[:512])[0]
        
        # 影响分析
        impact = "中性"
        if "涨" in content or "利好" in content or "增长" in content:
            impact = "利好"
        if "跌" in content or "利空" in content or "下降" in content:
            impact = "利空"
        if "政策" in content or "支持" in content:
            impact = "政策利好" if impact == "利好" else impact
        
        return {
            "标题": news["title"],
            "来源": news["source"],
            "链接": news["link"],
            "情感": result['label'],
            "置信度": f"{result['score']*100:.1f}%",
            "影响": impact,
            "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        print(f"分析新闻出错: {str(e)}")
        return None

def save_results(results):
    """保存分析结果"""
    with open('analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("分析结果已保存到 analysis_results.json")

def main():
    """主程序"""
    while True:
        try:
            news_items = fetch_finance_news()
            if not news_items:
                print("未获取到新闻，60秒后重试...")
                time.sleep(60)
                continue
                
            results = []
            print("\n最新财经新闻分析结果:")
            print("=" * 80)
            
            for news in news_items:
                analysis = analyze_news(news)
                if analysis:
                    results.append(analysis)
                    print(f"标题: {analysis['标题']}")
                    print(f"来源: {analysis['来源']}")
                    print(f"情感: {analysis['情感']} ({analysis['置信度']})")
                    print(f"影响: {analysis['影响']}")
                    print(f"链接: {analysis['链接']}")
                    print(f"分析时间: {analysis['分析时间']}")
                    print("-" * 80)
            
            if results:
                save_results(results)
            
            print(f"本次分析完成，{len(results)}条新闻已分析。下次更新在1小时后...")
            time.sleep(3600)  # 1小时更新一次
            
        except KeyboardInterrupt:
            print("\n程序已停止")
            break
        except Exception as e:
            print(f"主程序出错: {str(e)}，30秒后重试...")
            time.sleep(30)

if __name__ == "__main__":
    main()