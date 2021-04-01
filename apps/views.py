from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from django.http.request import QueryDict
from apps.serializers import TextSerializer
import json
import pandas as pd


from pickle import load

class SkyView(APIView):    
    
    parser_classes = (MultiPartParser, FormParser)
    def post(self, req, *args, **kwargs):
        
        
       
        # add new data to QueryDict(for mapping with Text, JSON)
        esti = req.data['Estimation']
        start = req.data['Start_Date']
        expe = req.data['Expect_Duration']
        equipList = req.data['Equipment_List']
        
        equipString = ''.join(equipList)
        
        new_data = req.data.dict()
        
        new_data['category_id'] = esti+start+expe+equipString
        
        new_query_dict = QueryDict('', mutable=True)
        new_query_dict.update(new_data)
        
        serializer = TextSerializer(data=new_query_dict)            
        
                
        
        #Equipment_List 입력 JSON parsing
        equip_dict=json.loads(equipList)
        print(equip_dict)
        print("========================")
        
        #Equipment_List 딕셔너리를 Equipment_List 데이터프레임으로 변경
        equip_series = pd.DataFrame(equip_dict)
        print(equip_series)
        print("========================")
        
        #Equipment_List 데이터프레임 특정 열 추출
        equip_amt = equip_series.loc[:,['amount']]
        print(equip_amt)
        print("========================")   
        
        #Estimation 데이터 Equipment_List  데이터프레임에 병합
        equip_amt.loc['esti'] = esti
        print(equip_amt)
        print("========================")          
        
        #행 열 바꾸기
        equip_data = equip_amt.transpose()                 
        
        
        
        #column 순서 변경
        cols = ['esti']+ [col for col in range(len(equip_data.columns)-1)]
        new_equip_data=equip_data[cols]
        print(new_equip_data)
        print("========================")
        
        #new_equip_data를 List 형식으로 변경
        equip_list = new_equip_data.values.tolist()
        new_equip = pd.DataFrame(equip_list, columns =[' Construction_Scale ','Equipment1', 
                                                       'Equipment2','Equipment3','Equipment4', 
                                                       'Equipment5', 'Equipment6','Equipment7', 
                                                       'Equipment8','Equipment9', 
                                                       'Equipment10','Equipment11', 
                                                       'Equipment12','Equipment13','Equipment14',
                                                       'Equipment15', 'Equipment16','Equipment17'] )
       
        
        #테스트 데이터 파일(sky_test_input1.csv)을 데이터프레임 형식으로 불러오기 
        data = pd.read_csv('/usr/src/app/apps/sky_test_input1.csv')
        print(data)
        print("========================")
        
        #테스트 데이터 파일과 new_equip_data를 병합
        new_raw_data = data.append(new_equip,ignore_index=True)
        print("new_raw_data : ",new_raw_data)
        
        #병합된 테스트 데이터 파일 AI prediction model 적용
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(equip_data)
        scaled_features = scaler.transform(equip_data)
        #print("scaled_features :", scaled_features)
        print("========================")
        
        
        new_raw_data.columns

        raw_data = new_raw_data[[' Construction_Scale ','Equipment1', 'Equipment2',     
                             'Equipment3','Equipment4', 'Equipment5', 'Equipment6',             
                             'Equipment7', 'Equipment8','Equipment9', 'Equipment10', 
                             'Equipment11', 'Equipment12','Equipment13', 'Equipment14', 
                             'Equipment15', 'Equipment16','Equipment17']]
        raw_data_completion= new_raw_data[[' Construction_Scale ']]
      
        #Import standardization functions from scikit-learn

        from sklearn.preprocessing import StandardScaler

        #Standardize the data set

        scaler = StandardScaler()

        scaler.fit(raw_data)

        scaled_features = scaler.transform(raw_data)
       
        scaled_data = pd.DataFrame(scaled_features, columns = raw_data.columns)

        #Standardize the data set

        scaler = StandardScaler()

        scaler.fit(raw_data_completion)

        scaled_features = scaler.transform(raw_data_completion)

        scaled_data_completion = pd.DataFrame(scaled_features, columns =                                
                                              raw_data_completion.columns)



        #Split the data set into training data and test data

        from sklearn.model_selection import train_test_split
        
        x = scaled_data
        y= scaled_data_completion
        x_training_data, x_test_data,= train_test_split(y, test_size = 0.2)

        model_basic = load(open('/usr/src/app/apps/sky_construction_Basic.pkl', 'rb'))
        predictions_basic = model_basic.predict(x)
        model_intermediate = load(open('/usr/src/app/apps/sky_construction_Intermediate.pkl', 'rb'))
        predictions_intermediate = model_intermediate.predict(x)
        model_advanced = load(open('/usr/src/app/apps/sky_construction_Advanced.pkl', 'rb'))
        predictions_advanced = model_advanced.predict(x)
        
        #예측값 각각의 변수에 저장
        basic_tail = predictions_basic[-1]
        intermediate_tail = predictions_intermediate[-1]
        advanced_tail = predictions_advanced[-1]
        print(basic_tail)
        print(intermediate_tail)
        print(advanced_tail)
        
        
        #예측값을 반환하는 함수
        def resultmodule():
            return {
                "Beginner" : basic_tail,
                "Intermediate" : intermediate_tail,
                "Advanced" : advanced_tail
                }             
        
        #REST API에 반환        
        if serializer.is_valid(): 
            result = resultmodule()  
            return Response(result, status=202)
        
        else:
            return Response(serializer.errors, status=400)
    




