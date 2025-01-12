# Ray Tracing Tutorial in Python

The original content is [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).
I reimplemented the tutorial code using Python.

## Chapters

[1 Creating an image](./01_creating_an_image.ipynb) 챕터를 제외한 다른 챕터들은 multi-processing과 numpy의 broadcasting을 이용해 렌더링 속도들 개선한 코드들이 추가로 구현되어 있습니다.

* [1 Creating an image](./01_creating_an_image.ipynb)
  * numpy array를 이용해 간단한 이미지를 생성하고 matplotlib을 이용해 시각화 합니다.

* [2 Camera and ray](./02_camera_and_ray_base.ipynb)
  * 카메라 파라미터를 정의하고 이를 이용해 카메라에서 이미지 평면으로 향하는 ray들을 계산합니다.
* [2-2 Camera and ray with numpy broadcasting](./02_camera_and_ray_nb.ipynb)

* [3 Adding a sphere](./03_adding_a_sphere_base.ipynb)
  * 장면에 구(sphere)를 추가하고 ray가 구와 만나는지 확인합니다.
* [3-2 Adding a sphere with numpy broadcasting](./03_adding_a_sphere_nb.ipynb)

* [4 Surface normal](./04_surface_normal_base.ipynb)
  * 장면에 ray와 만나는 구의 지점에서 구 표면의 법선(normal) 벡터를 계산합니다.
* [4-2 Surface normal with numpy broadcasting](./04_surface_normal_nb.ipynb)

* [5 Hittable object](./05_hittable_object_base.ipynb)
  * 여러 물체들을 장면에 배치하고 함께 ray가 어느 물체를 먼저 만나는지 확인합니다.
* [5-2 Hittable object with numpy broadcasting](./05_hittable_object_nb.ipynb)

* [6 Antialiasing](./06_antialiasing_base.ipynb)
  * 하나의 픽셀당 여러 ray를 샘플링해서 이미지의 계단(aliasing) 효과를 줄입니다.
* [6-2 Antialiasing with numpy broadcasting](./06_antialiasing_nb.ipynb)
* [6-3 Antialiasing with torch](./06_antialiasing_torch.ipynb)

* [7 Diffuse Material](./07_diffuse_material_base.ipynb)
  * 빛의 난반사(diffuse)를 구현합니다.
* [7-2 Diffuse Material with numpy broadcasting](./07_diffuse_material_nb.ipynb)
* [7-3 Diffuse Material with torch](./07_diffuse_material_torch.ipynb)

* [8 Metal](./08_metal_base.ipynb)
  * 메탈(metal) 재질을 구현합니다.
* [8-2 Metal with numpy broadcasting](./08_metal_nb.ipynb)
* [8-3 Metal with torch](./08_metal_torch.ipynb)

* [9 Dielectrics](./09_dielectrics_base.ipynb)
  * 유전체(dielectrics) 재질을 구현합니다.
* [9-2 Dielectrics with numpy broadcasting](./09_dielectrics_nb.ipynb)
* [9-3 Dielectrics with torch](./09_dielectrics_torch.ipynb)
