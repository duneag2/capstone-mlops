# Capstone MLOps 기반 머신러닝 워크플로우 자동화 프로젝트

MLOps(Machine Learning Operations)란 데이터 수집부터 인공지능 모델을 학습하고 배포하는 단계까지 머신러닝 워크플로우 전 과정에 대한 자동화된 파이프라인을 제공하는 시스템입니다.


[MLOPs for MLE](https://mlops-for-mle.github.io/tutorial/docs/intro)을 참고하여, 어쩌고 저쩌고..


프로젝트를 수행하는 방법은 다음과 같다.


## Data Prepare
먼저, 실험에 필요한 데이터를 준비하자. 본 프로젝트에서는 [cargo dataset](https://www.kaggle.com/datasets/morph1max/definition-of-cargo-transportation)을 사용했으나, 원하는 데이터셋을 적용시켜 사용할 수 있다.


데이터의 경로는 `./api_serving` 폴더의 하위 폴더로 만들어야한다. 본 프로젝트에서는 Original Image와 Corrupted Image를 각각 `./api_serving/monday`, `./api_serving/tuesday`라는 이름의 폴더로 만들어 사용하였다.
해당 이미지 파일은 [구글 드라이브](https://drive.google.com/drive/folders/1wvGfvN4Jc_7aVIzL5URvHUn57jLxV_Sq?usp=sharing)에서 다운받아 사용할 수 있다.


Corrupted Image 데이터셋인 `tuesday`는 `./data_prepare`에 있는 `image_corruption.py`를 이용해 만드는 것이 가능하다.




* 실행위치: `./data_prepare`
  ```
  python image_corruption.py -i ../api_serving/monday/background -o ../api_serving/tuesday/background --type mixed
  ```
  - `-i`와 `-o`는 각각 input folder와 output folder의 경로를 의미하며, path를 바꿔가며 모든 이미지 파일을 생성해주면 된다.
  - `--type`에는 `random_boxes`, `mosaic`, `random_line`, `mixed`가 있으며, 그 중 `mixed`는 나머지 세가지 옵션 중 하나로 무작위 선택되어 섞이게 한 것이다.
  - 본 프로젝트에서는 mixed 옵션을 사용하였다.


Corrupted Image의 예시는 다음과 같다.
<p align="center">
  <img src="https://github.com/duneag2/capstone-mlops/assets/137387521/139c9714-47b2-4376-b743-ab26eae04046" alt="tree64" width="250"/>
  <img src="https://github.com/duneag2/capstone-mlops/assets/137387521/a8fd5493-a53a-414f-a7f8-cf8385898688" alt="tree50" width="250"/>
  <img src="https://github.com/duneag2/capstone-mlops/assets/137387521/0be9b07d-4c7e-4a12-9e8f-d92415aa0fbf" alt="tree138" width="250"/>
</p>

<p align="center">
  <em>Random Boxes</em> | <em>Mosaic</em> | <em>Random Line</em>
</p>




또한, 본격적인 MLOPs 개발에 앞서 이미지 파일의 목록을 만들어야 한다. `./data_prepare/image_list.py`를 실행시키면 된다.
* 실행위치: `./data_prepare`
  ```
  python image_list.py
  ```
  - 이미지 파일 경로를 수정하기 위해서는 main 함수를 수정하면 된다.


## Data Generate

* 실행위치: `./data_generate`
  ```
  docker compose up -d --build --force-recreate
  ```
  ![data_generate 1](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F14519646-54d7-4a91-94c1-750beec3e8f9%2F310cbc04-24d5-4252-ab69-2bccab5ce52a%2FUntitled.png?table=block&id=c2d1c13f-5288-4e94-b274-bd89e4434816&spaceId=14519646-54d7-4a91-94c1-750beec3e8f9&width=2000&userId=54861078-d95f-4d03-8c5e-bd24b43177a5&cache=v2)
  ```
  docker ps
  ```
  ![data_generate 2](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F14519646-54d7-4a91-94c1-750beec3e8f9%2F182f3996-5914-48f0-afe1-f792211a6381%2FUntitled.png?table=block&id=205f7aea-f48e-4161-a9de-53502d1738b3&spaceId=14519646-54d7-4a91-94c1-750beec3e8f9&width=2000&userId=54861078-d95f-4d03-8c5e-bd24b43177a5&cache=v2)
  ```
  docker network ls
  ```
  ![data_generate 3](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F14519646-54d7-4a91-94c1-750beec3e8f9%2Ff37631c4-a691-461c-a8a9-4a2658a2fa6c%2FUntitled.png?table=block&id=f3a50026-d3b0-44cb-8da9-166a83f1f533&spaceId=14519646-54d7-4a91-94c1-750beec3e8f9&width=2000&userId=54861078-d95f-4d03-8c5e-bd24b43177a5&cache=v2)
  ```
  psql -h localhost -p 5432 -U myuser -d mydatabase
  ```
  - password: `mypassword`
  ![data_generate 4](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F14519646-54d7-4a91-94c1-750beec3e8f9%2Ffc4844d8-ee00-482a-a277-19378993a49e%2FUntitled.png?table=block&id=14004eb0-7ee3-4ea0-867f-d7071b7d0dbd&spaceId=14519646-54d7-4a91-94c1-750beec3e8f9&width=2000&userId=54861078-d95f-4d03-8c5e-bd24b43177a5&cache=v2)
  ```
  mydatabase=# select * from cargo order by id desc;
  ```
  ![data_generate 5](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F14519646-54d7-4a91-94c1-750beec3e8f9%2F0081b91a-55b8-49ac-9544-09be8c0ddb9a%2FUntitled.png?table=block&id=46984ef1-9173-478d-8995-70bbb8f58089&spaceId=14519646-54d7-4a91-94c1-750beec3e8f9&width=2000&userId=54861078-d95f-4d03-8c5e-bd24b43177a5&cache=v2)


## Model Registry
* 실행위치: `./model_registery`
  ```
  docker compose up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f51d472d-3748-406c-b65c-664c7a8cf310)




  [localhost:5001](http://localhost:5001/) 접속 (딱히 아무것도 안 뜬다면 정상)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ac82e811-0ed8-4b86-b287-537e045b9e0f)


  [localhost:9001](http://localhost:9001/) 접속 (username: `minio`, password: `miniostorage`)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/e4d6ad20-c912-4b6c-a9d9-b6b70dc8e0e7)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b6bdf68a-5243-48de-a331-336661b4e4c1)
  처음 들어가면 bucket이 없을 수 있다. -> Create a bucket -> 이름 `mlflow`로 설정 후 생성 (아래 토글들은 클릭하지 않으시면 됨)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/43c2f4c9-9cce-4087-891a-bcbb483a1106)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/7cac725f-50f1-49cb-9946-1ef7ce19b486)




  ```
  python3 save_model_to_registry.py --model-name "sk_model"
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f7902f4d-6c9b-4eee-bb8e-b1916641ffba)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/97ccc54d-e30a-4b25-bd7a-56646c177214)




  [localhost:5001](http://localhost:5001/) 접속 -> new_exp id 복사 `--run-id`로 입력 후 실행
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/4907c4f4-c974-49cc-9ad7-f88bb3196510)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/1f30c100-dd29-44f4-b280-a63c6e942920)
  ```
  python3 load_model_from_registry.py --model-name "sk_model" --run-id 70b965be026d4e5fb33cf6eccaa43b90
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d99fd613-7389-4cd6-a0d0-052ea8ffefd1)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ea0733c8-8187-40be-8173-48122e83c4bf)
  Metrics와 동일하게 나오는 것을 확인 할 수 있음

여기까지의 내용은 monday dataset 만으로 모델을 훈련시킨 과정이다. 여기서 훈련시킨 `sk-model`을 이용해 이후의 과정을 진행하였다. 별도로 tuesday dataset을 이용한 실험도 진행하였으나 이 내용은 맨 아래 **Training a model using both monday and tuesday dataset** 챕터에서 상술하였다.


## API Serving

* 실행위치: `./api_serving`
  [localhost:5001](http://localhost:5001/)에 있는 RUN ID 복붙해서 넣고 아래 코드 실행
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0e93c086-7137-4c4f-ace6-06ea0daff99d)
  ```
  python3 download_model.py --model-name sk_model --run-id 70b965be026d4e5fb33cf6eccaa43b90
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/a12632a4-fdb9-45cc-bfa0-f5e6d214afa3)




  ```
  uvicorn app:app --reload
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/22838778-f479-4602-8751-6b3389a25b9c)
  [http://localhost:8000/docs](http://localhost:8000/docs) 접속
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/291397d8-4f5f-4f69-b1d9-9d7cdb04031e)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/30831132-ce9c-4087-afef-d36b868780f6)
  `Try it out` 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/b3fdbecd-02ba-4447-9e16-42709cdc77c2)
  `"string"` 대신 `"background/background_13.jpg"`를 입력 후 Execute
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/96fae359-ac59-4247-8ee4-d254b3a50470)
  `target body`에 `0`(background를 의미)이 잘 나옴




  한번 더 해보자… `"concrete/concrete29.jpg"`를 입력하고 Execute
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d68c7f01-62d9-4342-8be9-2a2fbbe5da48)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ea44eb51-528b-4145-b58d-fc40d2e58005)
  `target 2`로 잘 나옴
  (참고로 background: 0 / brick: 1 / concrete: 2 / ground: 3 / wood: 4)




  ```
  docker compose up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c5fbdee4-3914-42d5-b15a-db5d627bd4d0)
  ```
  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"image_path": "background/background_13.jpg"}'
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/900969e6-513b-4c4b-92da-9ab93cef46f9)
  마찬가지로 target이 0으로 잘 나옴


## Kafka

* 실행위치: `./api_serving`
  ```
  docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d --build --force-recreate
  ```
  안되면 `bash`에서 exit한 다음 위 명령문 실행하고 다시 `bash`로 들어가서 `docker compose -p part7-kafka -f kafka-docker-compose.yaml up -d`하면 보통 된다.
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5493c236-3892-4fdb-9b24-104dbd6591f6)
  



  ```
  curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @source_connector.json
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5410612a-f58d-43d8-9e88-d0f951f3b647)
  curl문 안되는 경우, 도커 올린 직후라서 그럴수도 있음 좀 이따가 다시하면 될 수도 있음
  그래도 안되는 경우는 part7-kafka 도커 중 connect 도커가 exited 된것은 아닌지 확인해본다.
  만약 꺼졌다면 zookeeper → broker → schema → connect 순으로 켜보면 될 수 도 있음.
  `part7-kafka` 도커 삭제, `sudo docker system prune -a`로 캐시 삭제 하고 도커 다시 올려보면 될 수도 있다.




  ```
  curl -X GET http://localhost:8083/connectors
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/60ca7a33-e3a1-45c5-b318-cfbab5739da8)
  ```
  kafkacat -L -b localhost:9092
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/abac259e-6f66-4577-afe4-402791585a66)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5ddeb69c-c542-4bca-95d5-d5beda477ded)
  중간에 `topic "postgres-source-cargo"`에 해당하는 부분이 나오는지 확인해준다.




  ```
  kafkacat -b localhost:9092 -t postgres-source-cargo
  ```
  ![root@CHPCJ4_ _mnt_c_Users_USERSPC_capstone-mlops_kafka 2023-12-18 15-24-22](https://github.com/duneag2/capstone-mlops/assets/137387521/d8de0041-b5d1-4c3b-b977-3fa7596f8704)
  실시간 업데이트가 반영되고 있는 것을 확인할 수 있다.




  ```
  docker compose -p part7-target -f target-docker-compose.yaml up -d
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/ab2eceb3-402c-4f7a-823c-17b41a586d9b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f84fc07e-d4e3-473f-a506-0d5e2ab1b522)
  table-creator는 동작 후 exited 되는 것이 정상




  ```
  curl -X POST http://localhost:8083/connectors -H "Content-Type: application/json" -d @sink_connector.json
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/793a09b6-cd1f-4f41-9fd0-473bd4812f97)
  ```
  curl -X GET http://localhost:8083/connectors
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/204c92c9-c200-4fdb-b9e8-d229fdac6e45)
  ```
  curl -X GET http://localhost:8083/connectors/postgres-sink-connector
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c9d9d7a9-a2a3-40f6-807e-8f5a3b34c017)
  ```
  psql -h localhost -p 5433 -U targetuser -d targetdatabase
  ```
  - password: `targetpassword`
  ```
  SELECT * FROM cargo LIMIT 100;
  ```

## Dashboard Stream

* 실행위치: `./dashboard_stream`

  ```
  docker compose -p part8-stream -f stream-docker-compose.yaml up -d --build --force-recreate
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/59821c6a-cf21-48d9-82ee-d02a4d4cae19)
  ```
  psql -h localhost -p 5433 -U targetuser -d targetdatabase
  ```
  - password: `targetpassword`
  ```
  SELECT * FROM cargo_prediction LIMIT 100;
  ```
  ```
  docker compose -p part8-dashboard -f grafana-docker-compose.yaml up -d
  ```
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/fa1e069e-ff34-43ed-af54-cefe35e051ad)




  [localhost:3000](http://localhost:3000/) 접속 (id: `dashboarduser`, pw: `dashboardpassword`)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0c7b7eca-32db-4746-9821-5fb098ab22ee)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/1aff2a2c-7e6d-4614-9eda-a2d8c28b67c6)
  좌측상단 三자 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f81b1bd1-dd05-4ede-9b83-e688b19d49ba)
  Dashboard
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/9ad328f2-e0d0-4be7-b5d1-8b3a6f7e2155)
  Create Dashboard
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/4667f7fd-4355-43a1-b015-85f08d2034fb)
  우측 상단 톱니바퀴(⚙️) 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/f4de5851-4c85-4803-a355-3e6e2f42036a)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/656e9133-f633-46fc-9e31-ae2a61fd77c4)
  - Title: `Cargo classification`
  - Auto refresh: `1s,` 추가
  - `Refresh live dashboards` 토글 클릭하여 활성
  - 우측상단 `Save dashboard`




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/00ccb2a2-5e86-44ac-928d-3d49e3eb2266)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0f388425-c34a-4d30-ad65-2c02d6cf008f)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0541b598-6807-4a2f-b022-6620001a042b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5498e5d5-baa5-4784-a910-d088df7e178b)
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/0fe8b0eb-8507-4d89-a323-40c7693eea9a)
  `Default` 토글 해제
  - Name : `Inference-database`
  - Host : `target-postgres-server:5432`
  - Database : `targetdatabase`
  - User : `targetuser`
  - Password : `targetpassword`
  - TLS/SSL Mode : `disable`
  - Version : `14.0`
  `Save & test` 버튼 클릭 → Database Connection OK 문구가 뜸




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/8f762f9f-e961-4e48-bbd9-e344b7637887)
  다시 좌측 상단 三자 → Dashboard → Cargo classification
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/cc4d1910-cb9e-45bb-a9de-bacea4db2b39)
  Add visualization
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/745b292a-7a23-4b71-a955-8fc817acd6c6)
  Inference-database 클릭
  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/d9c881c5-6a66-4699-89ea-04e1ab2ba21d)
  - 우측 탭 : 패널의 이름, 차트의 종류 등을 설정합니다.
    - 기본 값으로 `Time series` 차트가 설정되어 있습니다.
    - 오른쪽 탭의 `Title` 에 패널의 이름을 붙여줍니다.
    - 이번 챕터에서는 `Cargo inference result` 로 설정하겠습니다.
  - 하단 탭 : 데이터 베이스에서 시각화할 테이블 및 열 정보를 설정합니다.
      - Data source : `Inference-database`
      - Table : `cargo_prediction`
      - Column : ➕ 버튼을 눌러 시각화 대상의 column 을 추가합니다.
      - `timestamp`
      - `target`
      - `Run query` 버튼 오른쪽의 `Code` 버튼을 클릭하고 `Limit` 부분을 지워줍니다.
  - `Run query` 버튼을 클릭합니다.
  우측상단의 Apply 하면 됨




  ![image](https://github.com/duneag2/capstone-mlops/assets/137387521/c13fc87f-dc26-4c38-ad85-76b534bdca42)
  우측상단 Last 6 hours라고 써있는 부분 클릭해서 From 부분을 now-30s로 수정, 우측상단 새로고침 버튼 눌러서 1s로 설정
  
  ![dashboard - Clipchamp로 제작](https://github.com/duneag2/capstone-mlops/assets/137387521/6502f6d4-b815-4c13-bb4c-abef076dd52d)


  잘 실행되는 것을 확인할 수 있다.


## Training a model using both monday and tuesday dataset

본 프로젝트에서는 ~~~~~~~~~~~~~~이유를 위해 어쩌고저쩌고를 했다.

아래의 내용은 Model Registery까지 진행 후 실행하면 된다.

* 실행위치: `./model_registry`
  ```
  python3 save_model_to_registry_tuesday.py --model-name "sk_model"
  ```
  ```
  python3 save_model_to_registry_random.py --model-name "sk_model"
  ```
  ```
  python3 save_model_to_registry_cp_l1norm.py --model-name "sk_model"
  ```
  ```
  python3 save_model_to_registry_cp_l2norm.py --model-name "sk_model"
  ```
  ```
  python3 save_model_to_registry_cp_cosine.py --model-name "sk_model"
  ```

각각 실행할 수 있으며, 실행 결과는 다음과 같았다.
![image](https://github.com/duneag2/capstone-mlops/assets/137387521/5389eb29-3295-4478-ac85-d23e8f90eb31)










  
