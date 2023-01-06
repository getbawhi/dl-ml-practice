fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


import numpy as np


#column_stack() 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결합니다 / 연결할 리스트는 파이썬의 튜플로 전달합니다
#np.column_stack(([1,2,3],[4,5,6]))

fish_data = np.column_stack((fish_length, fish_weight))
#print(fish_data[:5])

#np.concatenate 함수는 첫번째 차원을 따라 배열을 연결한다
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42)


#도미와 빙어가 잘 섞였는지 테스트 데이터 출력으로 확인
#print(test_target)
# -> 결과를 보니 도미 데이터로 편향되어 있음 (샘플링 편향)
#stratify를 사용하여 해결 할 것임

train_input, test_input, train_target, test_target = train_test_split(fish_data,fish_target, stratify = fish_target, random_state = 42)
#print(test_target)

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

#점검할 데이터를 넣고 출력하니 0이 나왔음 무엇이 문제인지 확인하자
#print(kn.predict([[25,150]]))

# import matplotlib.pyplot as plt
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker = '^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

#이웃까지의 거리와 이웃 샘플의 인덱스를 반환하는 kn.kneighbors() 함수
distances, indexes = kn.kneighbors([[25,150]])

# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker = '^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
# plt.xlabel = ('length')
# plt.ylabel = ('weight')
# plt.show()

#두 표를 비교해서 보면 첫 표만 볼 때에 도미 데이터와 가까우나 실제로 이웃을 확인해보면 빙어에 가깝게 나오는 것을 알 수 있음
#distances를 보아 확인하기로 함 
#print(distances)
# 결과를 보니 y축으로 조금만 멀어져도 아주 큰 값으로 계산된 것임

#x축 범위를 동일하게 맞추어 확인해 보겠음

# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(25, 150, marker = '^')
# plt.scatter(train_input[indexes,0], train_input[indexes,1], marker = 'D')
# plt.xlim((0,1000))
# plt.show()

#결과를 보니 y축만 반영이 된 것을 알 수 있음, x축과 y축의 스케일이 다르기 때문임 
#이를 위해 [데이터 전처리]를 해주어야 함

#표준점수 방법 
# 분산 : 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구한다
# 표준편차 : 분산의 제곱근, 데이터가 분산된 정도를 나타냄 
# 표준점수 : 각 데이터가원점에서 몇 표준점차만큼 떨어져 있는지를 나타내는 값

# 방법 : 평균을 빼고 표준편차를 나누어 주면 된다

# 평균과 표준편차 메소드로 구하기
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)

train_scaled = (train_input-mean)/std

# plt.scatter(train_scaled[:,0], train_scaled[:,1])
# plt.scatter(25, 150, marker = '^')
# plt.show()

#결과를 보니 예상과 다름, 이는 샘플을 동일한 비율로 변환하지 않았기 때문임
# 훈련 세트의 mean , std를 이용해 변환해야 한다

new = ([25, 150]-mean)/std
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0], new[1], marker = '^')
plt.show()

#이제 이 데이터 셋으로 다시 훈련

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std

kn.score(test_scaled, test_target)

print(kn.predict(([new])))

distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker = '^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker = 'D')
plt.show()
