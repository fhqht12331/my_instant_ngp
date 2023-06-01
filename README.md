# my_instant_ngp
아나콘다3에서 python path{your_instant-ngp_path}\scripts\colmap2nerf.py --run_colmap --의 형태로 작업을 지정

scripts 폴더와 images 폴더는 같은 폴더 내부에 있는 것이 편한 듯 함

--colmap2nerf.py 정의된 명령어--

--video_in : ffmpeg를 이용하여 영상 파일의 이름을 넣어 그 영상을 이미지로 변환
--video_fps : fps 지정하여 영상을 자름
--time_slice : 영상 전체 중 일부만 ffmpeg로 자르게 함
--run_colmap : images 폴더의 사진을 이용하여 colmap 실행
--colmap_matcher : colmap 실행 시 이미지 간의 좌표를 맞춰주는 matcher의 방법을 선택 - "exhaustive","sequential","spatial","transitive","vocab_tree"
--colmap_db : colmap의 db 파일의 이름 지정
--colmap_camera_model : colmap 실행 시 사용할 카메라의 학습 모델 선택 - "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"
--colmap_camera_params : 카메라 학습 모델에 따라 사용할 이미지의 고유 정보값 지정
--images : 저장할 이미지 경로 지정
--text : colmap 텍스트 파일의 입력 경로( --run_colmap을 사용하는 경우 자동으로 설정됨)
--aabb_scale : aabb_scale(instant-ngp 내 학습을 진행할 공간 크기 bounding box) 지정, 2^n 값의 자연수로 입력 (1, 2, 4, ... , 128)
--skip_early : 시작 시 필요 없는 이미지 건너뛰기
--keep_colmap_coords : COLMAP의 원래 참조 프레임에 transforms.json을 유지(미리 보기 및 렌더링을 위해 장면의 방향을 바꾸고 재배치할 수 없음)
--out : output 경로 지정
--vocab_path : vocab_tree의 학습 모델 경로 지정
--overwrite : 경로 내 기존 colmap 데이터 덮어쓰기를 물어보지 않음
--mask_catagories : 학습 이미지의 물체에 카테고리 마스크를 생성
