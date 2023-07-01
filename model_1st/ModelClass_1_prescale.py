    def prescaler(self):
        
        # 자료형보고 슬라이싱 (예를 들면, 첫 번째 column이 그냥 숫자 1, 2, 3...일 경우)
        data = data.iloc[:,1:10]  # 10은 임의의 값으로 자료의 최대 column

        # MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = pd.DataFrame(scaler.transform(data), columns = data.columns)
        # 여기까지 다 끝나고 잘 됐는지 확인하려면 model.isna()로 결측값 확인가능 (결측값이 있는 칸에만 False가 나타남)
        
        # 정규화가 필요한 경우
        data = pd.get_dummies(data)   # 데이터 columns 별로 정리하고, 나머지는 결측값 처리
        data = data.fillna(data.mean())  # 결측값에 model.mean(), 즉 평균값을 넣음으로써 그거보다 작고 크고 정규화가 가능함

        # 상관도 분석 (히트맵으로 한번 돌려봐야 알듯?)
        data_corr = data.corr()
        cols_train = []             # 상관도 분석해서 상관있는 변수들만 넣어주면 됩니다. 'capitalization' 이런식으로
        X_train_pre = data[cols_train]  # 상관있는 변수들만 잘라주고


        # Train data와  Test data 나누기 (일단 7:3으로 했음)
        answer = data['close'].values    # 배열에 들어갈 값은 주식의 종가(정답으로 체크할 변수) 'close' 같은 변수 1개 넣어주면 됨
        self.Train_data, self.Test_data = train_test_split(X_train_pre, answer, test_size = 0.3)

        
        pass