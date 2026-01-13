from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

with sync_playwright() as p:
   browser = p.chromium.launch()
   page = browser.new_page()
   page.goto(URL, wait_until="networkidle")

   html = page.content()

   with open("streamlit_report.html", "w", encoding="utf-8") as f:
       f.write(html)

   browser.close()
 