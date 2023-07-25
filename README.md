# HuBMAP - Hacking the Human Vasculature
@ Kaggle competition 

# Exploratory Data Analysis

### Description

---

The competition's goal is to design a model that can **segment** or **identify instances of microvascular structures** (like capillaries, arterioles, and venules) in 2D histology images from healthy **human kidney tissue slides**. This automated segmentation of microvasculature will enhance researchers' comprehension of blood vessel arrangement in human tissues.

## DATA

---

## Data Description

Quick Exploratory Data Analysis for [HuBMAP: Hacking the Kidney](https://www.kaggle.com/c/hubmap-kidney-segmentation) challenge

The HuBMAP data used in this hackathon includes 11 fresh frozen and 9 Formalin Fixed Paraffin Embedded (FFPE) PAS kidney images. Glomeruli FTU annotations exist for all 20 tissue samples; some of these will be shared for training, and others will be used to judge submissions.

- Words
    - TIFF
    
    TIFF(Tagged Image File Format)는 고해상도 이미지를 저장하기 위해 사용되는 파일 형식입니다. 이 파일 형식은 특히 이미지 조작, 그래픽 디자인, 스캔, 인쇄 등과 같은 환경에서 많이 사용됩니다.
    
    TIFF 파일은 비트맵 데이터를 저장하며, 이는 이미지가 픽셀로 이루어진 점 배열이라는 의미입니다. 이 형식은 손실 없는 압축을 지원하여 이미지의 품질이 저하되지 않도록 보장하며, 이는 JPEG 등의 손실 압축 형식과는 대조적입니다.
    
    또한 TIFF는 여러 페이지의 이미지를 하나의 파일에 저장할 수 있어서, 문서 스캔에 주로 사용됩니다. 또한 메타데이터를 저장할 수 있어서, 이미지에 대한 추가 정보(예: 제작자, 촬영 일시, 카메라 설정 등)를 파일에 포함시킬 수 있습니다.
    
    그러나 TIFF 파일은 고해상도와 무손실 압축 때문에 파일 크기가 커질 수 있어서, 웹에서는 이를 렌더링하는 데 시간이 오래 걸릴 수 있습니다. 따라서 웹에서는 주로 더 작은 파일 크기를 가진 JPEG 또는 PNG 형식이 사용됩니다.
    
    - **포름알데히드 고정 파라핀 포맷 (FFPE)** 
    포름알데히드로 조직을 고정하고 파라핀 왁스러 묻혀서 보존하는 방법 : 보통 조직 절단하여 관찰
    - **글로머룰리** 
    신장의 일부로, 신장 기능을 수행하는 미세한 필터링 구조를 말한다.
    - **FTU**는 "Functional Tissue Unit"의 약자로, 생물학적 기능을 수행하는 조직의 기본 단위를 가리킨다. 따라서 신장의 글로머룰리는 신장의 FTU 중 하나라고 볼 수 있다.
    

### Train masks

- train.csv
contrains the unique IDs for each image, as well as an RLE-encoding scheme.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1fce9c3e-71b7-4e73-8427-576e2eb70af5/Untitled.png)

- Run Length Encoding (RLE) 
RLE is a simple form of lossless data compression that runs on sequences with the sam value occurring many consecutie times. It encodes the sequence to store only a single value and its count. (ex AAAABBBCCD → 4A3B2C1D)
- Number of samples
Number of train images : 15
Number of test images : 5
- Number of files
Number of train files : 7003 (tif file)
Number of test files : 1 (tif file)
- HuBMAP-20-dataset_information.csv
contains additional information (including anonymized patient data) about each image.

## Evaluation

---

[Open Images Challenge Visual Relationships Detection evaluation](https://www.notion.so/Open-Images-Challenge-Visual-Relationships-Detection-evaluation-b3ac5b8df6654043b8b84cba939f6ef4?pvs=21)
