# data/

이 폴더에는 수집 스크립트(`scripts/`)만 들어 있습니다. 영상·채굴 크롭·AI-Hub 파생 자산은 저장소에 넣지 않았습니다.
코드가 기대하는 배치는 아래와 같습니다(경로는 `src/`, `scripts/` 의 코드가 그대로 씁니다).

```
data/
├── scripts/                      수집·변환 스크립트 (저장소에 포함)
├── backgrounds/                  배경 장면 (1024×1024 @0.3 m JPEG)
│   ├── gsi/ naip/ linz/ ign/ eri/    fetch_bg_v2.py fetch --source <소스>
│   ├── scenes_1024/                  이전에 받은 46장면(NAIP 40·ERI 6). 받은 스크립트는 이 저장소에 없음
│   ├── _meta/                        좌표 후보·소스별 장부
│   ├── splits/v2_train.txt           학습 3,679장 (make_splits_v2.py)
│   ├── splits/v2_test.txt            시험 317장 (어떤 모델도 학습에 안 씀)
│   ├── mined_negatives_v2/           v2 모델의 실배경 오탐 크롭 4,667장 (scripts/train_final.sh mine_v2)
│   ├── mined_negatives_v4/           v4 모델의 실배경 오탐 크롭 11,722장 (scripts/train_final.sh mine_v4, 2차 분류기용)
│   └── pdok_tidal/                   네덜란드 갯벌 장면 (scripts/verifier/fetch_pdok.py, 2차 분류기용)
├── aihub_71363/                  AI-Hub 71363 (2차 분류기용, 직접 내려받기)
│   ├── download/…                    라벨 JSON zip (build_aihub_debris.py 의 DL 경로)
│   └── derived/                      real03/train · sprites/train · kr_neg/train · kr_bg/train
└── aihub_506_sample/             AI-Hub 506 샘플 (2차 분류기용, 직접 내려받기)
    ├── 라벨링데이터/ 원천데이터/
    └── derived/sprites/              extract_506_sprites.py
```

## 출처와 라이선스

| 데이터 | 쓰임 | 출처 | 라이선스 |
|---|---|---|---|
| USGS NAIP 정사영상 (0.3 m 원본만) | 배경 1,100장면 + 기존 40장면 | imagery.nationalmap.gov | Public Domain |
| NOAA ERI 평상시(Pre_Event) 영상 | 배경 350장면 + 기존 6장면 | registry.opendata.aws/noaa-eri | NOAA NODD 공개 데이터 |
| LINZ 항공 정사영상 (0.2~0.3 m) | 배경 750장면 | registry.opendata.aws/nz-imagery | CC BY 4.0 |
| 일본 국토지리원 정사사진 타일 `ort` | 배경 950장면 | maps.gsi.go.jp | 국토지리원 콘텐츠 이용규약 (PDL1.0 호환) |
| IGN BD ORTHO HR | 배경 800장면 | geoservices.ign.fr | Licence Ouverte / Etalab 2.0 |
| PDOK Luchtfoto 2024 | 2차 분류기 음성 (갯벌) | pdok.nl | CC BY 4.0 |
| AI-Hub 71363 「AI기반 국립공원 변화탐지 모니터링 플랫폼 구축」 태안 드론 영상 | 2차 분류기 음성·실측 양성 | aihub.or.kr | AI-Hub 이용정책 (원본 재배포 금지) |
| AI-Hub 506 「해안 오염물질 데이터」 샘플 | 2차 분류기 양성 합성용 조각 | aihub.or.kr | AI-Hub 이용정책 (원본 재배포 금지) |
| OSM 해안선 (coastlines-split-4326) | 배경 장면 중심 좌표 뽑기에만 사용, 저장·학습 안 함 | osmdata.openstreetmap.de | ODbL |

- ERI 는 태풍 직후 영상이 아니라 평상시 영상을 썼습니다. 실제 잔해가 음성으로 학습되는 것을 피하려는 선택입니다.
- AI-Hub 데이터는 주최 측 공지(외부 공개 데이터 학습 사용 가능, AI-Hub 약관 준수)에 따라 2차 분류기 학습에만 썼습니다.
- 채굴 크롭은 시트로 눈으로 확인해 쓰레기로 보이는 것이 없음을 확인한 뒤 음성으로 썼습니다.
