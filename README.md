# Dacon_stock_price_prediction

# 제2회 KRX 주식 투자 알고리즘 경진대회

## 알고리즘 | 정형 | 시계열 | 포트폴리오 구성 | 금융 | 샤프 지수

- 상금 : 5,000 만원
    
    [data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTAuMzM1IiBjeT0iMTAuMzM1IiByPSI3LjU4NSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KPHBhdGggZD0iTTguMDA0NDMgNi4xMjIzNkw5LjE1MzQxIDE0LjM4ODNMNy44ODc5NiAxNC4zODgyTDYuNzk0MDkgNi4xMjIzMkw4LjAwNDQzIDYuMTIyMzZaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMTIuNjY1NSA2LjEyMjM2TDExLjUxNjUgMTQuMzg4M0wxMi43ODIgMTQuMzg4MkwxMy44NzU4IDYuMTIyMzJMMTIuNjY1NSA2LjEyMjM2WiIgZmlsbD0id2hpdGUiLz4KPHBhdGggZD0iTTkuNzgwNTUgNi44NDMxOEw4LjYzMTU3IDE0LjM4ODJMOS44OTcwMiAxNC4zODgyTDEwLjk5MDkgNi44NDMxNEw5Ljc4MDU1IDYuODQzMThaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMTEuMTIxOCA2Ljg0MzE4TDEyLjI3MDggMTQuMzg4MkwxMS4wMDUzIDE0LjM4ODJMOS45MTE0NSA2Ljg0MzE0TDExLjEyMTggNi44NDMxOFoiIGZpbGw9IndoaXRlIi8+CjxyZWN0IHg9IjYuMDA5NzciIHk9IjkuNjA0IiB3aWR0aD0iOC42NDk5IiBoZWlnaHQ9IjEuMDgxMjQiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHZpZXdCb3g9IjAgMCAyMCAyMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMTAuMzM1IiBjeT0iMTAuMzM1IiByPSI3LjU4NSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjUiLz4KPHBhdGggZD0iTTguMDA0NDMgNi4xMjIzNkw5LjE1MzQxIDE0LjM4ODNMNy44ODc5NiAxNC4zODgyTDYuNzk0MDkgNi4xMjIzMkw4LjAwNDQzIDYuMTIyMzZaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMTIuNjY1NSA2LjEyMjM2TDExLjUxNjUgMTQuMzg4M0wxMi43ODIgMTQuMzg4MkwxMy44NzU4IDYuMTIyMzJMMTIuNjY1NSA2LjEyMjM2WiIgZmlsbD0id2hpdGUiLz4KPHBhdGggZD0iTTkuNzgwNTUgNi44NDMxOEw4LjYzMTU3IDE0LjM4ODJMOS44OTcwMiAxNC4zODgyTDEwLjk5MDkgNi44NDMxNEw5Ljc4MDU1IDYuODQzMThaIiBmaWxsPSJ3aGl0ZSIvPgo8cGF0aCBkPSJNMTEuMTIxOCA2Ljg0MzE4TDEyLjI3MDggMTQuMzg4MkwxMS4wMDUzIDE0LjM4ODJMOS45MTE0NSA2Ljg0MzE0TDExLjEyMTggNi44NDMxOFoiIGZpbGw9IndoaXRlIi8+CjxyZWN0IHg9IjYuMDA5NzciIHk9IjkuNjA0IiB3aWR0aD0iOC42NDk5IiBoZWlnaHQ9IjEuMDgxMjQiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=)
    
- 2023.07.03 ~ 2023.07.28 23:59 - 예선

**Dataset Info.**

- **train.csv [파일]**
2021년 6월 1일부터 2023년 5월 30일까지의 일별 시세정보
- **sample_submission.csv [파일] - 제출 양식 (세부사항은 [링크](https://dacon.io/competitions/official/236117/talkboard/408729?page=1&dtype=recent) 참고)**
종목코드 : 각 주식종목의 고유 코드
순위 : 거래행위 결정을 위한 수익률 순위
1~200위는 매수 (Long)
1801위부터 2000위는 공매도 (Short)

- 핵심 → 단순 수익율이 아닌 Risk 대비 수익

- 평가기간에 대한 규정
- a) Public
- 대회 Public 기간 중 예측시점은 2023년 5월 30일
- 참가자들을 예측시점으로부터 향후 15일의 주식 거래일 동안 포트폴리오에 따라 주식거래를 진행한다고 가정함
- 해당 기간은 2023년 5월 31일부터 6월 21일이며, 주식거래가 이뤄지지 않는 날은 제외
- 
    
    !https://dacon.s3.ap-northeast-2.amazonaws.com/attach/talkboard/236117/459664/168831687546687.png
    
- Public 기간 중 예측시점보다 미래의 정보를 활용하는 것은 예견편향성에 해당되며, 이는 규칙 위반에 해당
- b) Private
- 대회 Private 기간 전 예측시점은 2023년 7월 28일, 즉 대회 종료시점
- 참가자들을 예측시점으로부터 향후 15일의 주식 거래일 동안 포트폴리오에 따라 주식거래를 진행한다고 가정함
- 해당 기간은 2023년 7월 31일부터 8월 21일이며, 주식거래가 이뤄지지 않는 날은 제외
- 
    
    !https://dacon.s3.ap-northeast-2.amazonaws.com/attach/talkboard/236117/459664/1688317294563433.png
    
- 이때 역시 예측시점보다 미래의 정보를 활용하는 것은 **예견편향성**에 해당되며, 이는 규칙 위반에 해당
- 다만, Private 기간이 시작되기 전 예측시점까지의 데이터가 제공될 예정
- 참가자들은 해당 데이터가 제공된 이후 Private 기간을 위한 포트폴리오를 제작
- 해당 포트폴리오 제작을 위해서는 예측시점(2023년 7월 28일)까지의 데이터 활용가능
- Private 리더보드는 8월 3일부터 공개되기 시작하며, 8월 21일의 샤프 지수로 최종 순위가 결정됨 (이는 수식상 n이 1과 2일때 샤프 지수가 계산되지 않기 때문)

## Baseline

- Prophet 
2000개 종목 개별 평가 - 
- 특징 : close 기준으로 target / 첫 값이 안 맞는 문제
→
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9a287bb8-13c4-4a35-9b72-aaae5fafa368/Untitled.png)
    
- First value - 첫값을 맞춘 prophet
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a6334b32-71f6-4ce1-b9ae-3d58ba4931e4/Untitled.png)
    

- RandomForestregressor - target값을 close 값이 아닌 sharpe ratio 값으로 지정
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f4a4a4f-ec58-4762-ae65-ae6a17c386f3/Untitled.png)
    
- LGBMRegressor - target값을 sharpe ratio 값으로 지정 + Feature engineering
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c554d9b3-342d-47ff-827f-f5b1dbf41278/Untitled.png)
    
- 그 외 시도 - trend값으로 예측 / 변화율로 예측 
⇒ 결과 bad

- LSTM / Transformer 적용
