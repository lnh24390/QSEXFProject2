# QSEXFProject2
## 주제 :  AI 이미지 기반쓰레기 분리배출 안내
### 팀명 : 물시느끝
설명 : 물음표로 시작해서 느낌표로 끝난다.
## 팀원
이남호(팀장), 여동엽, 오승재, 김아인
## 데이터셋은 저작권으로 인하여 직접 사이트 회원가입 하여 다운로드 받으시길 바랍니다.
dataset은 로보플로우에서 다운로드하시면 되겠습니다.  
  
## 텐서플로우와 파이토치 차이
GPU 사용 여부 CUDA연동
텐서플로우보다 파이토치가 GPU 호환성이 좋음  
GPU는 병렬연산처리로 연산이 빠르고 CPU는 한번에 처리할 수 있는 데이터가 적기에 GPU로 데이터 처리가 용이한 Pytorch사용  
CPU만으로 빠른속도 학습 시도시 Overflow 발생가능성이 있음 그리고 데이터 처리량이 방대해질수록 학습시간이 오래걸릴 수 밖에 없음 그러므로  
안전하면서 빠른학습을 위해 pytorch를 사용


## 데이터 라벨링 과정
라벨링 툴을 만든 후 로보플로우에서 라벨링이 미흡하게된 부분을 마우스로 라벨링 작업
## 전이학습 이전과 이후 차이
비대상 객체가 포함된 이미지로 전이학습했지만 기존 탐지 대상의 미탐도 늘어, 전이학습 이전 모델이 더 나은 결과를 보였다.  
* 전이 학습 이전의 학습 및 카메라 실행 코드  
[https://github.com/lnh24390/QSEXFProject2/blob/main/dataset%20traing%20code_Commentary.py
](https://github.com/lnh24390/QSEXFProject2/blob/main/code/dataset%20traing%20code_Commentary.py)  
* 전이학습 코드  
https://github.com/lnh24390/QSEXFProject2/tree/main/trash_detection_project/03_transfer_learning
## 시각화 자료
<img width="1588" height="728" alt="image" src="https://github.com/user-attachments/assets/9043eaef-b0b6-4e44-bab8-f9298483d2fc" />  
<img width="1588" height="728" alt="image" src="https://github.com/user-attachments/assets/203980da-718c-4ba6-9e05-a35eaac9bddc" />

## APP 설치법  
https://github.com/lnh24390/QSEXFProject2/releases/tag/app-v1.0.2 에서 다운로드 후
설정 → 보안 및 개인정보 보호 → 보안위험 자동차단(OFF상태 전환) → 내 파일에서 APK파일 터치 후 무시하고 설치 → 설치 후 설정 → 보안 및 개인정보 보호 → 보안위험 자동차단(ON상태으로 전환)
## 어플 시연
