# Capstone MLOPs 제목 수정해야할라나??
본 프로젝트는 어쩌구저쩌구 어라왈와ㅣ 호ㅑㅗㄷ ㅓㅜㅏㅇ낭랑 ㅏㅣㅗ댜 ㅗ두 대충 abstract에 해당하는 내용 어쩌고 저쩌고 롸롸롸롸뢀 


[MLOPs for MLE](https://mlops-for-mle.github.io/tutorial/docs/intro)을 참고하여, 어쩌고 저쩌고..


프로젝트를 수행하는 방법은 다음과 같다.


## Data Prepare
먼저, 실험에 필요한 데이터를 준비하자. 본 프로젝트에서는 [cargo dataset](https://www.kaggle.com/datasets/morph1max/definition-of-cargo-transportation)을 사용했으나, 원하는 데이터셋을 적용시켜 사용할 수 있다.


데이터의 경로는 `./api_serving` 폴더의 하위 폴더로 만들어야한다. 본 프로젝트에서는 Original Image와 Corrupted Image를 각각 `./api_serving/monday`, `./api_serving/tuesday`라는 이름의 폴더로 만들어 사용하였다.
해당 이미지 파일은 [구글 드라이브]()에서 다운받아 사용할 수 있다.


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
