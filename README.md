# Ray Tracing Tutorial in Python

The original content is [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
I just reimplemented the tutorial code using Python.

## Chapters
* [01 Creating an image](./01_creating_an_image.ipynb)
  * numpy array를 이용해 간단한 이미지를 생성하고 matplotlib을 이용해 시각화 합니다.
* [02 Camera and ray](./02_camera_and_ray_base.ipynb)
  * 카메라 파라미터를 정의하고 이를 이용해 카메라에서 이미지 평면으로 향하는 ray들을 계산합니다.
* [02 Camera and ray opt](./02_camera_and_ray_opt.ipynb)
  * 02 Camera and ray에서 사용되는 python의 for loop을 제거하고 numpy의 broadcasting을 이용해 계산 속도를 개선합니다. 
* [03 Adding a sphere](./03_adding_a_sphere_base.ipynb)
  * 장면에 구(sphere)를 추가하고 ray가 구와 만나는지 확인합니다.
* [03 Adding a sphere opt](./03_adding_a_sphere_opt.ipynb)
  * 03 Adding a sphere에서 사용되는 python의 for loop을 제거하고 numpy의 broadcasting을 이용해 계산 속도를 개선합니다. 
* [04 Surface normal](./04_surface_normal_base.ipynb)
  * 장면에 ray와 만나는 구의 지점에서 구 표면의 법선(normal) 벡터를 계산합니다.
* [04 Surface normal opt](./04_surface_normal_opt.ipynb)
  * 04 Surface normal에서 사용되는 python의 for loop을 제거하고 numpy의 broadcasting을 이용해 계산 속도를 개선합니다. 