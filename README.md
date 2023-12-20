# Capstone MLOPs 제목 수정해야할라나??
본 프로젝트는 어쩌구저쩌구 어라왈와ㅣ 호ㅑㅗㄷ ㅓㅜㅏㅇ낭랑 ㅏㅣㅗ댜 ㅗ두 대충 abstract에 해당하는 내용 어쩌고 저쩌고 롸롸롸롸뢀 


[MLOPs for MLE](https://mlops-for-mle.github.io/tutorial/docs/intro)을 참고하여, 어쩌고 저쩌고..


프로젝트를 수행하는 방법은 다음과 같다.


## Data Prepare
먼저, 실험에 필요한 데이터를 준비하자. 본 프로젝트에서는 [cargo dataset](https://www.kaggle.com/datasets/morph1max/definition-of-cargo-transportation)을 사용했으나, 원하는 데이터셋을 적용시켜 사용할 수 있다.


데이터의 경로는 `./api_serving` 폴더의 하위 폴더로 만들어야한다. 본 프로젝트에서는 Original Image와 Corrupted Image를 각각 `./api_serving/monday`, `./api_serving/tuesday`라는 이름의 폴더로 만들어 사용하였다.
해당 이미지 파일은 [구글 드라이브]()에서 다운받아 사용할 수 있다.


Corrupted Image 데이터셋인 `tuesday`는 `./data_prepare`에 있는 `image_corruption.py`를 이용해 만드는 것이 가능하다.


