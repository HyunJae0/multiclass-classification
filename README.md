# 2023년 제3회 금융 데이터 활용 아이디어 경진대회
## 팀 프로젝트
제가 맡은 부분은 전처리부터 모델링까지입니다.

여러 개의 데이터를 결합하여 사용할 때 발생하는 결측치를 단순 평균값이나 중앙값 등 하나의 값으로 결측치를 채우는 단순 대치법(Single Imputation)이 아닌, 다중 대치법(Multiple Imputation)을 이용해 채우고, 과적합이 발생하는 base model에 과적합 억제 기법을 적용하여 다중 분류 모델(Multiclass Classification Model)을 구현했습니다.

예측에 사용한 데이터는 '롯데 카드'에서 제공한 '전라북도 행정동 업종별 상권 평가' 데이터와 '소상공인 마당'에서 제공하는 읍면동별 3km를 기준 '직장인구의 상반기 평균 소득, 평균 소비 금액과 하반기 평균 소득, 평균 소비 금액'을 크롤링한 데이터를 결합한 데이터입니다.

## Stack
```
Python
R
Keras
TensorFlow
```

# 코드 실행
## 1. Handling Missing Values
### 1.1 Multivariate Imputation by Chained Equations(MICE)
아래 .R로 기술되어 있음 https://github.com/HyunJae0/multiclass-classification/blob/main/%EA%B8%88%EC%9C%B5%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B3%B5%EB%AA%A8%EC%A0%84%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EA%B2%B0%EC%B8%A1%EA%B0%92%20%EC%B2%98%EB%A6%AC.R

