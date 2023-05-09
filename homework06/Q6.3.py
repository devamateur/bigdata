from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import datetime

def current_weather(result):
        weather_url = 'https://www.weather.go.kr/weather/observation/currentweather.jsp'
        print(weather_url)
        html = urllib.request.urlopen(weather_url)  # url을 연다
        soupWeather = BeautifulSoup(html, 'html.parser')
        tag_tbody = soupWeather.find('tbody')  # 웹페이지의 tbody 태그를 찾음
        for sido in tag_tbody.find_all('tr'):   # tbody 태그 안의 tr 태그를 찾음
            if len(sido) <= 3:
                break
            weather_td = sido.find_all('td')   # tr 태그 안의 td 태그를 찾
            sido_gu = weather_td[0].string   # 0번째 td태그(시도구)
            degree = weather_td[5].string    # 5번째 td태그(온도)
            humid = weather_td[10].string     # 10번째 td태그(습도)
            result.append([sido_gu]+[degree]+[humid])
        return

#[CODE 0]
def main():
    result = []
    print('Current weather crawling >>>>>>>>>>>>>>>>>>>>>>>>>>')
    current_weather(result)   #[CODE 1] 호출 
    weather_tbl = pd.DataFrame(result, columns=('sido-gu', '온도','습도'))
    weather_tbl.to_csv('weathers.csv', encoding='cp949', mode='w', index=True)
    
    del result[:]
    

if __name__ == '__main__':
    main()
    
# 만든 csv 파일을 읽어와서 데이터프레임으로 
weather_result = pd.read_csv('weathers.csv', encoding='cp949')
