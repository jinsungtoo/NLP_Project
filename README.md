# NLP_Project

기간 : 2023.03.06 - 2023.03.16


## 목적 
AIhub 데이터를 이용하여 NLP 모델을 학습한 결과 AI가 어느정도까지 감정을 인식하고 텍스트를 통해 사람의 심리를 얼마나 헤아릴 수 있는지 결과를 예상해보고 최종구현 결과와 비교하여 학습이 잘 되었는지 웹을 통해 서비스 제공한다.

## 데이터 소개
> AIHub 오픈 데이터 사용

    https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86


![image](https://user-images.githubusercontent.com/115756142/226775924-d184a048-cfd4-4eea-a970-35b4e2797029.png)
- JSON 형식의 라벨링 데이터와 xlsx 형식의 원천 데이터로 구성
(csv로 데이터를 다루기 위해 원천데이터를 사용)
***
![image](https://user-images.githubusercontent.com/115756142/226778848-80a4bd1a-c92b-4d62-a507-bbb2f391b989.png)
- 6가지 대분류 속 각각 9개의 소분류 감정 라벨로 구성되어 있지만 라벨별 데이터 부족으로 대분류 라벨만 사용

## 데이터 현황
감정 상태를 나타내는 첫번째 발화만 데이터로 사용

train 데이터 : 51630개

valid 데이터 : 6641개 

총 약 58000개의 데이터 사용

## 학습 과정

> 1) 오픈 데이터 다운로드 및 데이터 재구성

> 2) NLP 감정 분석 코드 학습

>> 1. colab 환경에서 코드 학습 후 windows 환경에서 학습 

> 3) HTML / CSS를 사용하여 웹 구현

> 4) 학습된 결과와 웹 연동

> 5) 텍스트 입력 시 감정 분석 알림


## 결과 스크린샷
![image](https://user-images.githubusercontent.com/115756142/226574133-3820cb87-55d2-493f-ba76-9fd81221b788.png)


## 구현 결과
![test1](https://user-images.githubusercontent.com/115756142/226573798-e3ec8549-2e8c-4c79-b171-a9a3ac3196d1.gif)
