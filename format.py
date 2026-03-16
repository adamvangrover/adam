from bs4 import BeautifulSoup
import bs4

def analyze_dashboard():
    with open('showcase/dashboard.html') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    for d in soup.find_all('div'):
        if d.has_attr('className') and d['className'] == 'min-h-screen p-6':
            print("Found min-h-screen p-6")

    for m in soup.find_all('main'):
        print(f"Main tag: {m}")

analyze_dashboard()
