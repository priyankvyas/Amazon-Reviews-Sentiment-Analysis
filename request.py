import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from urllib3.util import Retry

url = ('https://www.amazon.ca/Sunstar-888JC-GUM-Advanced-Flossers/dp/B074Q537B9/ref=zg_bs_c_beauty_d_sccl_3/131'
       '-6244951-0884666?pd_rd_w=vWLFc&content-id=amzn1.sym.6f50da21-510d-4da6-9182-ff71d80f9354&pf_rd_p=6f50da21'
       '-510d-4da6-9182-ff71d80f9354&pf_rd_r=6GBNSGBD9CT0XQ0EAB53&pd_rd_wg=qz7Wq&pd_rd_r=6402a0a9-4ada-4e83-bf44'
       '-219736037deb&pd_rd_i=B074Q537B9&th=1')
product_url = url.split('/')[:4]
product_id_url = url.split('/')[5:6]
reviews_url = ""
for s in product_url:
    reviews_url += s + '/'
reviews_url += 'product-reviews/' + product_id_url[0] + '/' + ("ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType"
                                                               "=all_reviews")

session = requests.session()
session.headers.update({
    "authority": "www.amazon.ca",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,"
              "application/signed-exchange;v=b3;q=0.7",
    "Referer": url,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 "
                  "Safari/537.36 Edg/120.0.0.0"})
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[503], allowed_methods={'GET'})
session.mount("https://", HTTPAdapter(max_retries=retries))
response = session.get(reviews_url, timeout=5)

if response.status_code == 200:
    parsed_response = BeautifulSoup(response.content, 'html.parser')
    reviews_list = parsed_response.find(id="cm_cr-review_list")
    reviews_title = reviews_list.find_all("a", "review-title")
    reviews_text = reviews_list.find_all("span", "review-text")
    # TODO: Add verification status as well.

    if reviews_list is not None:
        for review_title, review_text in zip(reviews_title, reviews_text):
            print(review_title.find("span", "a-icon-alt").text.split(' ')[0], review_title.find_all("span")[-1].text,
                  review_text.find("span").text)
else:
    print(response.status_code)
