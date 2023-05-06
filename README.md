# NLP_Project

기간 : 2023.03.06 - 2023.03.16


## 목적 
AIhub 데이터를 이용하여 NLP 모델을 학습한 결과 AI가 어느정도까지 감정을 인식하고 텍스트를 통해 사람의 심리를 얼마나 헤아릴 수 있는지 결과를 예상해보고 최종구현 결과와 비교하여 학습이 잘 되었는지 웹을 통해 서비스를 제공한다.

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

>> • colab 환경에서 코드 학습

>> • 이후 windows 환경에서 학습 

> 3) HTML / CSS를 사용하여 웹 구현

> 4) 학습된 결과와 웹 연동

> 5) 텍스트 입력 시 감정 분석 알림

## 나의 역할

    - 한글을 잘 인식할 수 있는 NLP 모델 탐색
    - 감정 분석 코드 학습
    - 가상환경 내 Kobert 모델 구현 방법 모색
    - 웹 사이트 CSS 코드 구현


## 문제점 및 보완할 점

1. Colab 환경에선 NLP 학습이 잘되었지만 window 환경에선 진행되지 않음.
    -> 확인 결과 Python 버전이 달라 호환성의 문제였고 버전을 다운그레이드 하여 해결


2. 웹 페이지를 꾸미는 과정에서 프론트엔드 언어가 익숙치 않아 어려움을 느낌.
    -> 복잡하지 않게 간단한 코드로 구현


## 결과 스크린샷
![image](https://user-images.githubusercontent.com/115756142/226574133-3820cb87-55d2-493f-ba76-9fd81221b788.png)


## 구현 결과
![test1](https://user-images.githubusercontent.com/115756142/226573798-e3ec8549-2e8c-4c79-b171-a9a3ac3196d1.gif)
